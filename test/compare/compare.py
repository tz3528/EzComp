#===-- compare.py ---------------------------------------------*- Python -*-===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

"""
比较测试结果脚本
1. 编译 reference.cpp (支持 O2/O3 两种优化级别)
2. 编译 ezcomp 的 .comp 文件（支持多种 pass-pipeline 配置）
3. 多次运行（默认10次），随机顺序执行，取平均值
4. 比较运行生成的 result.h5 与预期结果
5. 绘制性能比较图（显示相对于 O2 的加速比）
"""

import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np
import psutil

# 由 CMake 配置注入
EZCOMP_EXECUTABLE = "@EZCOMP_EXECUTABLE@"
COMPARE_TEST_SOURCE_DIR = "@COMPARE_TEST_SOURCE_DIR@"
CXX_COMPILER = "@CXX_COMPILER@"
HDF5_LIBS = "@EZCOMPUTE_HDF5_LIBS@"
HDF5_INCLUDE_DIRS = "@HDF5_INCLUDE_DIRS@"
OPENMP_LIBS = "@OPENMP_LIBS@"

NUM_RUNS = 10
CONFIG_FILE_NAME = "compare_config.json"

# ── 绘图常量 ──────────────────────────────────────────────────────────────────
BASELINE_LABEL = "Ref-O2"

# Tableau 10 配色方案
REF_FACE_COLOR = "#4E79A7"        # Reference 柱子（钢蓝）
EZCOMP_FACE_COLOR = "#F28E2B"     # EzComp 柱子（暖橙）
BAR_EDGE_COLOR = "#333333"        # 柱子边框色
BAR_EDGE_WIDTH = 0.5              # 柱子边框宽度

# 中文字体候选列表（仿宋GB2312），按平台优先级排列
_CN_FONT_FAMILIES = [
    "FangSong_GB2312",      # Linux (实际注册名)
    "仿宋_GB2312",           # Linux (中文名)
    "FangSong",             # Windows
    "仿宋",                  # Windows (中文名)
    "STFangsong",           # macOS
]


def load_configs():
    """
    从 compare 目录加载配置文件

    Returns:
        dict: 配置字典，包含 reference_configs 和 ezcomp_configs

    Raises:
        SystemExit: 配置文件不存在或配置无效时退出程序
    """
    config_path = os.path.join(COMPARE_TEST_SOURCE_DIR, CONFIG_FILE_NAME)

    if not os.path.exists(config_path):
        print(f"错误: 配置文件 {config_path} 不存在")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "ezcomp_configs" not in config:
            print("错误: 配置文件中缺少 'ezcomp_configs' 字段")
            sys.exit(1)

        if "reference_configs" not in config:
            print("错误: 配置文件中缺少 'reference_configs' 字段")
            sys.exit(1)

        return config

    except json.JSONDecodeError as e:
        print(f"错误: 配置文件 JSON 解析失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取配置文件失败: {e}")
        sys.exit(1)


def find_comp_file(test_dir):
    comp_files = glob.glob(os.path.join(test_dir, "*.comp"))
    if not comp_files:
        return None
    if len(comp_files) > 1:
        print(f"警告: 找到多个 .comp 文件，使用第一个: {comp_files[0]}")
    return comp_files[0]


def get_executable_path(output_dir, name):
    return os.path.join(output_dir, name + (".exe" if sys.platform == "win32" else ""))


def run_command(cmd, cwd=None, env=None):
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=run_env)


def compile_reference(compiler, ref_cpp, output_dir, opt_level, hdf5_libs, hdf5_include_dirs, openmp=False, openmp_libs=""):
    suffix = "_omp" if openmp else ""
    executable_path = get_executable_path(output_dir, f"reference_{opt_level}{suffix}")
    cmd = [compiler, f"-{opt_level}", "-o", executable_path, ref_cpp]
    
    # OpenMP 编译选项
    if openmp:
        cmd.append("-DUSE_OPENMP")
        cmd.append("-fopenmp")
    
    cmd += [arg for inc in filter(None, hdf5_include_dirs.split(";")) for arg in ("-I", inc)]
    cmd += [lib for lib in filter(None, hdf5_libs.split(";"))]
    
    # OpenMP 库路径
    if openmp:
        for lib in filter(None, openmp_libs.split(";")):
            cmd.append(lib)
            # 添加 rpath（仅 Linux 和 macOS）
            if sys.platform.startswith("linux") or sys.platform == "darwin":
                lib_dir = os.path.dirname(lib)
                if lib_dir:
                    cmd.append(f"-Wl,-rpath,{lib_dir}")

    try:
        result = run_command(cmd)
        if result.returncode != 0:
            print(f"[{opt_level}] 编译失败:")
            print(result.stderr)
            return False, None
        return True, executable_path
    except FileNotFoundError:
        print(f"错误: 找不到编译器: {compiler}")
    except Exception as e:
        print(f"编译时出错: {e}")
    return False, None


def parse_timer_output(stderr_text):
    times = {}
    labeled_pattern = r"\[TIMER\](?:\s+\[([^\]]+)\])?\s+([^:]+):\s+(\d+)h\s+(\d+)m\s+(\d+)s\s+(\d+)ms"
    simple_pattern = r"\[TIMER\]\s+(\d+)h\s+(\d+)m\s+(\d+)s\s+(\d+)ms"

    for match in re.finditer(labeled_pattern, stderr_text):
        _, label, hours, minutes, seconds, ms = match.groups()
        times[label.strip()] = f"{hours}h {minutes}m {seconds}s {ms}ms"

    if times:
        return times

    match = re.search(simple_pattern, stderr_text)
    if match:
        hours, minutes, seconds, ms = match.groups()
        times["default"] = f"{hours}h {minutes}m {seconds}s {ms}ms"
    return times


def parse_time_to_seconds(time_str):
    if not time_str or time_str == "N/A":
        return None
    match = re.match(r"(\d+)h\s+(\d+)m\s+(\d+)s\s+(\d+)ms", time_str)
    if not match:
        return None
    hours, minutes, seconds, ms = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + ms / 1000.0


def run_executable(executable, working_dir, args=None, runtime_env=None):
    # 解析 runtime_env 字符串，如 "OMP_NUM_THREADS=8 OMP_PROC_BIND=true"
    # 支持 OMP_NUM_THREADS=auto 自动使用物理核心数（避免超线程缓存争用）
    env_dict = {}
    if runtime_env:
        for item in runtime_env.split():
            if "=" in item:
                key, value = item.split("=", 1)
                if key == "OMP_NUM_THREADS" and value == "auto":
                    value = str(psutil.cpu_count(logical=False))
                env_dict[key] = value

    try:
        result = run_command([executable, *(args or [])], cwd=working_dir, env=env_dict if env_dict else None)
        if result.returncode != 0:
            print(f"运行失败: {executable}")
            print(result.stderr)
            return False, None

        times = parse_timer_output(result.stderr)
        return True, parse_time_to_seconds(times.get("计算用时") or times.get("default"))
    except FileNotFoundError:
        print(f"错误: 找不到可执行文件: {executable}")
    except Exception as e:
        print(f"运行时出错: {e}")
    return False, None


def compile_ezcomp(ezcomp_path, comp_file, working_dir, test_name, output_name, emit="compile", pipeline=None):
    cmd = [ezcomp_path, comp_file, f"-emit={emit}"]
    if pipeline:
        cmd.append(f"--pass-pipeline={pipeline}")

    print(f"编译命令: {' '.join(cmd)}")
    print(f"工作目录: {working_dir}")
    print("-" * 50)

    try:
        result = run_command(cmd, cwd=working_dir)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"ezcomp 返回错误码: {result.returncode}")
            return False, None
        print("编译成功")
    except FileNotFoundError:
        print(f"错误: 找不到 ezcomp 可执行文件: {ezcomp_path}")
        return False, None
    except Exception as e:
        print(f"运行 ezcomp 时出错: {e}")
        return False, None

    default_executable = get_executable_path(working_dir, test_name)
    if not os.path.exists(default_executable):
        print(f"错误: 找不到生成的可执行文件: {default_executable}")
        return False, None

    target_executable = get_executable_path(working_dir, output_name)
    if os.path.exists(target_executable):
        os.remove(target_executable)
    shutil.move(default_executable, target_executable)

    print(f"重命名: {default_executable} -> {target_executable}")
    print("=" * 50)
    return True, target_executable


def load_h5_result(file_path):
    if not os.path.exists(file_path):
        return None
    with h5py.File(file_path, "r") as f:
        return {"result": f["result"][:] if "result" in f else None, "datasets": list(f.keys())}


def verify_results(all_data, rtol=1e-6):
    ezcomp_data = all_data.get("ezcomp")
    if ezcomp_data is None:
        return False, ["ezcomp 结果不存在"]

    ezcomp_result = ezcomp_data.get("result")
    if ezcomp_result is None:
        return False, ["ezcomp 结果中没有 'result' 数据集"]

    ref_key = "O3"
    ref_data = all_data.get(ref_key)
    if ref_data is None:
        return False, [f"参考结果 {ref_key} 不存在"]

    ref_result = ref_data.get("result")
    if ref_result is None:
        return False, [f"参考结果 {ref_key} 中没有 'result' 数据集"]

    if ezcomp_result.shape != ref_result.shape:
        return False, [f"结果形状不匹配: ezcomp {ezcomp_result.shape} vs 参考 {ref_result.shape}"]

    errors = []
    nan_count = np.sum(np.isnan(ezcomp_result))
    inf_count = np.sum(np.isinf(ezcomp_result))
    if nan_count > 0:
        errors.append(f"ezcomp 结果包含 {nan_count} 个 NaN 值")
    if inf_count > 0:
        errors.append(f"ezcomp 结果包含 {inf_count} 个 Inf 值")
    if errors:
        return False, errors

    abs_diff = np.abs(ezcomp_result - ref_result)
    # ACM 风格相对误差: |差值| / max(1, |参考值|)
    # 当参考值较小时退化为绝对误差，避免除以接近0的值
    rel_diff = abs_diff / np.maximum(1.0, np.abs(ref_result))

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.nanmax(rel_diff)
    max_rel_idx = np.unravel_index(np.nanargmax(rel_diff), rel_diff.shape)

    print(f"\n结果验证:")
    print(f"  最大绝对误差: {max_abs_diff:.6e}")
    print(f"  最大相对误差: {max_rel_diff:.6e}")
    print(f"  最大相对误差位置: {max_rel_idx}")
    print(f"  该位置 ezcomp 值: {ezcomp_result[max_rel_idx]:.6e}")
    print(f"  该位置参考值: {ref_result[max_rel_idx]:.6e}")

    if max_rel_diff > rtol:
        return False, [f"最大相对误差 {max_rel_diff:.6e} 超过容差 {rtol}"]

    print("  ✓ 结果验证通过!")
    return True, []


# ═══════════════════════════════════════════════════════════════════════════════
#  绘  图
# ═══════════════════════════════════════════════════════════════════════════════


def _setup_fonts():
    """配置学术论文字体：英文 Times New Roman，数学 STIX"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"] + plt.rcParams["font.serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    })


def _get_cn_font(size=13):
    """获取中文宋体 FontProperties，自动在候选列表中查找可用字体"""
    from matplotlib.font_manager import findfont
    for family in _CN_FONT_FAMILIES:
        fp = FontProperties(family=family, size=size)
        resolved = findfont(fp, fallback_to_default=False)
        if resolved and "LastResort" not in resolved:
            return fp
    print("警告: 未找到宋体字体，中文标题将使用默认 serif 字体")
    return FontProperties(family="serif", size=size)


def plot_performance(perf_data, output_path="performance_comparison.pdf",
                     num_runs=NUM_RUNS):
    """
    绘制学术风格性能对比柱状图（Tableau 10 配色）

    - 白底，隐去顶/右轴线
    - Reference 钢蓝，EzComp 暖橙，无纹理
    - 柱顶标注时间值，x 轴下方标注相对加速比
    - 输出 PDF 矢量图 + PNG 预览
    """
    labels = perf_data["labels"]
    means = np.array(perf_data["times"])
    stds = np.array(perf_data.get("stds", [0.0] * len(labels)))
    n_ref = perf_data.get("n_ref", 0)
    n_ez = perf_data.get("n_ez", 0)

    if len(labels) == 0:
        print("没有足够的数据来绘制性能比较图")
        return

    # ── 字体与全局样式 ────────────────────────────────────────────────────
    _setup_fonts()
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 0.8,
    })
    cn_title_font = _get_cn_font(size=13)

    # ── 加速比计算 ────────────────────────────────────────────────────────
    baseline_idx = next(
        (i for i, l in enumerate(labels) if l == BASELINE_LABEL), 0
    )
    baseline_time = means[baseline_idx]
    speedups = baseline_time / np.where(means > 0, means, np.nan)

    # ── 图尺寸（单栏 ≈15 cm，按柱数自适应） ─────────────────────────────
    n_bars = len(labels)
    fig_w_inch = max(5.9, n_bars * 1.1 + 1.0)
    fig_h_inch = 3.8
    fig, ax = plt.subplots(figsize=(fig_w_inch, fig_h_inch))

    x = np.arange(n_bars)
    width = min(0.60, 3.0 / n_bars)

    # ── 绘制柱子（同组同色，无纹理） ─────────────────────────────────────
    for i in range(n_bars):
        ax.bar(
            x[i], means[i], width,
            color=REF_FACE_COLOR if i < n_ref else EZCOMP_FACE_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=3,
        )

    # ── 误差棒 ────────────────────────────────────────────────────────────
    ax.errorbar(
        x, means, yerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=4,
    )

    # ── 柱顶数值标注 ─────────────────────────────────────────────────────
    y_pad = (means + stds).max() * 0.02
    for i in range(n_bars):
        ax.text(
            x[i], means[i] + stds[i] + y_pad,
            f"{means[i]:.3f} s",
            ha="center", va="bottom",
            fontsize=9,
            color="#333333",
        )

    # ── x 轴：名称 + 加速比 ──────────────────────────────────────────────
    x_tick_labels = []
    for i, label in enumerate(labels):
        if i == baseline_idx:
            x_tick_labels.append(f"{label}\n(baseline)")
        else:
            x_tick_labels.append(f"{label}\n({speedups[i]:.2f}\u00d7)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels, fontsize=9)

    # ── y 轴 ─────────────────────────────────────────────────────────────
    ax.set_ylabel("计算时间 (s)", fontproperties=_get_cn_font(size=11),
                  labelpad=8)
    # 调整y轴范围以容纳误差棒
    y_max = (means + stds).max()
    ax.set_ylim(0, y_max * 1.15)

    # ── 标题 ─────────────────────────────────────────────────────────────
    ax.set_title(
        "性能对比",
        fontproperties=cn_title_font,
        pad=12,
    )

    # ── 网格线（淡灰水平虚线） ───────────────────────────────────────────
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45,
                  color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # ── 隐去顶/右轴线 ────────────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── 图例 ──────────────────────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=REF_FACE_COLOR, edgecolor=BAR_EDGE_COLOR,
              label="Reference (C++)"),
        Patch(facecolor=EZCOMP_FACE_COLOR, edgecolor=BAR_EDGE_COLOR,
              label="EzComp"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        fontsize=9, framealpha=0.9, edgecolor="#cccccc",
    )

    # ── 保存 ──────────────────────────────────────────────────────────────
    plt.tight_layout()

    pdf_path = output_path
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    print(f"\n矢量图已保存至: {pdf_path}")

    png_path = os.path.splitext(output_path)[0] + ".png"
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight",
                pad_inches=0.08)
    print(f"预览图已保存至: {png_path}")

    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="比较测试结果")
    parser.add_argument(
        "test_dir",
        nargs="?",
        default=os.environ.get("COMPARE_TEST_DIR", ""),
        help="测试目录名 (也可通过环境变量 COMPARE_TEST_DIR 设置)",
    )
    parser.add_argument("--rtol", type=float, default=1e-6, help="相对容差 (默认: 1e-6)")
    parser.add_argument("--no-plot", action="store_true", help="不生成性能比较图")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help=f"运行次数 (默认: {NUM_RUNS})")
    args = parser.parse_args()

    if not args.test_dir:
        parser.error("需要指定测试目录名，用法: compare.py <test_dir> 或设置环境变量 COMPARE_TEST_DIR")

    test_dir_path = os.path.join(COMPARE_TEST_SOURCE_DIR, args.test_dir)
    if not os.path.isdir(test_dir_path):
        print(f"错误: 测试目录不存在: {test_dir_path}")
        return 1

    comp_file = find_comp_file(test_dir_path)
    if not comp_file:
        print(f"错误: 在目录 {test_dir_path} 中未找到 .comp 文件")
        return 1

    ref_cpp = os.path.join(test_dir_path, "reference.cpp")
    if not os.path.exists(ref_cpp):
        print(f"错误: 在目录 {test_dir_path} 中未找到 reference.cpp")
        return 1

    # 加载配置
    config = load_configs()
    reference_configs = config["reference_configs"]
    ezcomp_configs = config["ezcomp_configs"]

    print(f"测试目录: {args.test_dir}")
    print(f"Comp 文件: {comp_file}")
    print(f"Reference: {ref_cpp}")
    print(f"ezcomp: {EZCOMP_EXECUTABLE}")
    print(f"编译器: {CXX_COMPILER}")
    print(f"运行次数: {args.runs}")
    print("=" * 50)

    print("========== 编译 reference.cpp ==========")
    compiled_refs = {}
    for ref_config in reference_configs:
        opt_level = ref_config["opt_level"]
        label = ref_config["label"]
        key = ref_config["key"]
        openmp = ref_config.get("openmp", False)
        runtime_env = ref_config.get("runtime_env", "")
        success, exe_path = compile_reference(
            CXX_COMPILER, ref_cpp, os.getcwd(), opt_level, 
            HDF5_LIBS, HDF5_INCLUDE_DIRS,
            openmp=openmp, openmp_libs=OPENMP_LIBS
        )
        print(f"[{label}] 编译成功: {exe_path}" if success else f"[{label}] 编译失败")
        if success:
            compiled_refs[key] = {"exe": exe_path, "label": label, "opt_level": opt_level, "openmp": openmp, "runtime_env": runtime_env}
    print("=" * 50)

    print("========== 编译 EzComp ==========")
    compiled_ezcomps = {}
    for cfg in ezcomp_configs:
        print(f"--- {cfg['label']} ---")
        success, exe_path = compile_ezcomp(
            EZCOMP_EXECUTABLE,
            comp_file,
            os.getcwd(),
            args.test_dir,
            f"{args.test_dir}_{cfg['output_name']}",
            emit=cfg["emit"],
            pipeline=cfg.get("pipeline"),
        )
        if not success:
            print(f"[{cfg['label']}] 编译失败")
            return 1
        print(f"[{cfg['label']}] 编译成功: {exe_path}")
        compiled_ezcomps[cfg["key"]] = {"exe": exe_path, "label": cfg["label"], "runtime_env": cfg.get("runtime_env", "")}
    print("=" * 50)

    print(f"========== 运行性能测试 ({args.runs} 次) ==========")
    reference_keys = [cfg["key"] for cfg in reference_configs]
    ezcomp_keys = [cfg["key"] for cfg in ezcomp_configs]
    run_times = {key: [] for key in reference_keys + ezcomp_keys}

    for run_idx in range(args.runs):
        round_start = time.time()
        round_tasks = [("ref", key) for key in compiled_refs] + [
            ("ezcomp", key) for key in compiled_ezcomps
        ]
        random.shuffle(round_tasks)

        for task_idx, (task_type, version_key) in enumerate(round_tasks):
            if task_type == "ezcomp":
                label = compiled_ezcomps[version_key]["label"]
                runtime_env = compiled_ezcomps[version_key].get("runtime_env", "")
                success, run_time = run_executable(compiled_ezcomps[version_key]["exe"], os.getcwd(), runtime_env=runtime_env)
                if not success:
                    print(f"Round {run_idx + 1}: {label} 失败")
                    return 1
            else:
                ref_info = compiled_refs[version_key]
                runtime_env = ref_info.get("runtime_env", "")
                success, run_time = run_executable(
                    ref_info["exe"],
                    os.getcwd(),
                    [f"result_{ref_info['opt_level']}_{run_idx}.h5", f"{ref_info['opt_level']}_{run_idx}"],
                    runtime_env=runtime_env,
                )
                if not success:
                    print(f"Round {run_idx + 1}: {ref_info['label']} 失败")
                    return 1

            run_times[version_key].append(run_time)

        print(f"Round {run_idx + 1}/{args.runs} 完成，本轮用时: {time.time() - round_start:.3f}s")
    print("=" * 50)

    print("========== 统计结果 ==========")
    perf_data = {"labels": [], "times": [], "stds": [], "n_ref": 0, "n_ez": 0}

    for cfg in reference_configs:
        key = cfg["key"]
        if run_times.get(key):
            arr = np.array(run_times[key])
            avg, std = float(np.mean(arr)), float(np.std(arr))
            print(f"{cfg['label']}: 平均 {avg:.3f}s ± {std:.3f}s (n={len(arr)})")
            perf_data["labels"].append(cfg["label"])
            perf_data["times"].append(avg)
            perf_data["stds"].append(std)
            perf_data["n_ref"] += 1

    for cfg in ezcomp_configs:
        key = cfg["key"]
        if run_times.get(key):
            arr = np.array(run_times[key])
            avg, std = float(np.mean(arr)), float(np.std(arr))
            print(f"{cfg['label']}: 平均 {avg:.3f}s ± {std:.3f}s (n={len(arr)})")
            perf_data["labels"].append(cfg["label"])
            perf_data["times"].append(avg)
            perf_data["stds"].append(std)
            perf_data["n_ez"] += 1

    print("=" * 50)

    print("========== 验证结果 ==========")
    all_data = {}
    if (ezcomp_data := load_h5_result("result.h5")):
        all_data["ezcomp"] = ezcomp_data
        print(f"ezcomp result.h5: 数据集 {ezcomp_data['datasets']}")

    # 使用最后一个 reference 配置作为验证基准
    ref_config = reference_configs[-1]
    ref_key = ref_config["opt_level"]
    last_ref_file = f"result_{ref_key}_{args.runs - 1}.h5"
    if (ref_data := load_h5_result(last_ref_file)):
        all_data[ref_key] = ref_data
        print(f"{ref_config['label']} {last_ref_file}: 数据集 {ref_data['datasets']}")

    passed, errors = verify_results(all_data, rtol=args.rtol)
    if not passed:
        print("\n❌ 结果验证失败!")
        for err in errors:
            print(f"  错误: {err}")
        return 1
    print("=" * 50)

    if not args.no_plot:
        print("========== 绘制性能比较图 ==========")
        plot_performance(
            perf_data,
            output_path=os.path.join(os.getcwd(), "performance_comparison.pdf"),
            num_runs=args.runs,
        )
        print("=" * 50)

    print("========== 清理中间文件 ==========")
    cleaned_count = 0

    # 清理 reference 生成的中间文件
    for ref_cfg in reference_configs:
        for run_idx in range(args.runs):
            path = f"result_{ref_cfg['opt_level']}_{run_idx}.h5"
            if os.path.exists(path):
                os.remove(path)
                cleaned_count += 1

    # 清理 ezcomp 生成的 result.h5
    ezcomp_result = "result.h5"
    if os.path.exists(ezcomp_result):
        os.remove(ezcomp_result)
        cleaned_count += 1

    print(f"已清理 {cleaned_count} 个 h5 文件")
    print("=" * 50)

    print("\n✅ 所有测试通过!")
    return 0


if __name__ == "__main__":
    sys.exit(main())