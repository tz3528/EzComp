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
import matplotlib.pyplot as plt
import numpy as np

# 由 CMake 配置注入
EZCOMP_EXECUTABLE = "@EZCOMP_EXECUTABLE@"
COMPARE_TEST_SOURCE_DIR = "@COMPARE_TEST_SOURCE_DIR@"
CXX_COMPILER = "@CXX_COMPILER@"
HDF5_LIBS = "@EZCOMPUTE_HDF5_LIBS@"
HDF5_INCLUDE_DIRS = "@HDF5_INCLUDE_DIRS@"

EZCOMP_COLOR = "#e74c3c"
NUM_RUNS = 10
CONFIG_FILE_NAME = "compare_config.json"


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


def run_command(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def compile_reference(compiler, ref_cpp, output_dir, opt_level, hdf5_libs, hdf5_include_dirs):
    executable_path = get_executable_path(output_dir, f"reference_{opt_level}")
    cmd = [compiler, f"-{opt_level}", "-o", executable_path, ref_cpp]
    cmd += [arg for inc in filter(None, hdf5_include_dirs.split(";")) for arg in ("-I", inc)]
    cmd += [lib for lib in filter(None, hdf5_libs.split(";"))]

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


def run_executable(executable, working_dir, args=None):
    try:
        result = run_command([executable, *(args or [])], cwd=working_dir)
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
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(np.abs(ref_result) < 1e-10, abs_diff, abs_diff / np.abs(ref_result))

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


def plot_performance(perf_data, output_path="performance_comparison.png"):
    labels, times, colors = perf_data["labels"], perf_data["times"], perf_data["colors"]
    if not labels:
        print("没有足够的数据来绘制性能比较图")
        return

    baseline_time = next((t for label, t in zip(labels, times) if label == "Ref-O2"), None)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, times, color=colors, edgecolor="black", linewidth=1.2)

    ax.set_xlabel("Version", fontsize=12)
    ax.set_ylabel("Compute Time (seconds)", fontsize=12)
    ax.set_title("Performance Comparison: EzComp vs Reference (10 runs average)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    for bar, time_val, label in zip(bars, times, labels):
        height = bar.get_height()
        if baseline_time is not None and label == "Ref-O2":
            annotation = f"{time_val:.3f}s\n(baseline)"
        elif baseline_time is not None and baseline_time > 0:
            annotation = f"{time_val:.3f}s\n({baseline_time / time_val:.2f}x)"
        else:
            annotation = f"{time_val:.3f}s"

        ax.annotate(
            annotation,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max(times) * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor="#3498db", edgecolor="black", label="Reference (C++)"),
            Patch(facecolor=EZCOMP_COLOR, edgecolor="black", label="EzComp"),
        ],
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPerformance comparison chart saved to: {output_path}")
    plt.close()


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
        success, exe_path = compile_reference(CXX_COMPILER, ref_cpp, os.getcwd(), opt_level, HDF5_LIBS, HDF5_INCLUDE_DIRS)
        print(f"[{label}] 编译成功: {exe_path}" if success else f"[{label}] 编译失败")
        if success:
            compiled_refs[key] = {"exe": exe_path, "label": label, "opt_level": opt_level}
    print("=" * 50)

    print("========== 编译 EzComp ==========")
    compiled_ezcomps = {}
    for config in ezcomp_configs:
        print(f"--- {config['label']} ---")
        success, exe_path = compile_ezcomp(
            EZCOMP_EXECUTABLE,
            comp_file,
            os.getcwd(),
            args.test_dir,
            f"{args.test_dir}_{config['output_name']}",
            emit=config["emit"],
            pipeline=config.get("pipeline"),
        )
        if not success:
            print(f"[{config['label']}] 编译失败")
            return 1
        print(f"[{config['label']}] 编译成功: {exe_path}")
        compiled_ezcomps[config["key"]] = {"exe": exe_path, "label": config["label"]}
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

        for task_type, version_key in round_tasks:
            if task_type == "ezcomp":
                label = compiled_ezcomps[version_key]["label"]
                success, run_time = run_executable(compiled_ezcomps[version_key]["exe"], os.getcwd())
                if not success:
                    print(f"Round {run_idx + 1}: {label} 失败")
                    return 1
            else:
                ref_info = compiled_refs[version_key]
                success, run_time = run_executable(
                    ref_info["exe"],
                    os.getcwd(),
                    [f"result_{ref_info['opt_level']}_{run_idx}.h5", f"{ref_info['opt_level']}_{run_idx}"],
                )
                if not success:
                    print(f"Round {run_idx + 1}: {ref_info['label']} 失败")
                    return 1

            run_times[version_key].append(run_time)

        print(f"Round {run_idx + 1}/{args.runs} 完成，本轮用时: {time.time() - round_start:.3f}s")
    print("=" * 50)

    print("========== 统计结果 ==========")
    perf_data = {"labels": [], "times": [], "colors": []}
    for label, key, color in (
            *[(cfg["label"], cfg["key"], "#3498db") for cfg in reference_configs],
            *[(cfg["label"], cfg["key"], EZCOMP_COLOR) for cfg in ezcomp_configs],
    ):
        if run_times.get(key):
            times_arr = np.array(run_times[key])
            avg_time, std_time = np.mean(times_arr), np.std(times_arr)
            print(f"{label}: 平均 {avg_time:.3f}s ± {std_time:.3f}s (n={len(times_arr)})")
            perf_data["labels"].append(label)
            perf_data["times"].append(avg_time)
            perf_data["colors"].append(color)
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
        plot_performance(perf_data, output_path=os.path.join(os.getcwd(), "performance_comparison.png"))
        print("=" * 50)

    print("========== 清理中间文件 ==========")
    cleaned_count = sum(
        os.remove(path) is None
        for ref_cfg in reference_configs
        for run_idx in range(args.runs)
        for path in [f"result_{ref_cfg['opt_level']}_{run_idx}.h5"]
        if os.path.exists(path)
    )
    print(f"已清理 {cleaned_count} 个中间文件")
    print("=" * 50)

    print("\n✅ 所有测试通过!")
    return 0


if __name__ == "__main__":
    sys.exit(main())