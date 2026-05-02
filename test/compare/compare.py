#===-- compare.py ---------------------------------------------*- Python -*-===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

"""
比较测试结果脚本 - 调度模块
1. 编译 reference.cpp (支持 O2/O3 两种优化级别)
2. 编译 ezcomp 的 .comp 文件（支持多种 pass-pipeline 配置）
3. 多次运行（默认10次），随机顺序执行，取平均值
4. 比较运行生成的 result.h5 与预期结果
5. 绘制性能比较图（显示相对于 O2 的加速比）
"""

import argparse
import os
import random
import re
import sys
import time

import numpy as np

from io_handler import (
    load_configs,
    find_comp_file,
    compile_reference,
    compile_ezcomp,
    run_executable,
    load_h5_result,
    verify_results,
    cleanup_intermediates,
    # CMake 注入的常量
    EZCOMP_EXECUTABLE,
    COMPARE_TEST_SOURCE_DIR,
    CXX_COMPILER,
    HDF5_LIBS,
    HDF5_INCLUDE_DIRS,
    OPENMP_LIBS,
)
from plotter import plot_performance

NUM_RUNS = 10


# ─────────────────────────────────────────────────────────────────────────────
#  时间解析
# ─────────────────────────────────────────────────────────────────────────────


def parse_timer_output(stderr_text):
    """解析计时器输出，提取时间信息"""
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
    """将时间字符串转换为秒数"""
    if not time_str or time_str == "N/A":
        return None
    match = re.match(r"(\d+)h\s+(\d+)m\s+(\d+)s\s+(\d+)ms", time_str)
    if not match:
        return None
    hours, minutes, seconds, ms = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + ms / 1000.0


def extract_run_time(stderr_text):
    """从 stderr 中提取运行时间（秒）"""
    times = parse_timer_output(stderr_text)
    return parse_time_to_seconds(times.get("计算用时") or times.get("default"))


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

    # 创建输出目录
    output_dir = os.path.join(os.getcwd(), "compare")
    os.makedirs(output_dir, exist_ok=True)

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

    # ── 编译 reference.cpp ──────────────────────────────────────────────────────
    print("========== 编译 reference.cpp ==========")
    compiled_refs = {}
    for ref_config in reference_configs:
        opt_level = ref_config["opt_level"]
        label = ref_config["label"]
        key = ref_config["key"]
        openmp = ref_config.get("openmp", False)
        runtime_env = ref_config.get("runtime_env", "")
        success, exe_path = compile_reference(
            CXX_COMPILER, ref_cpp, output_dir, opt_level,
            HDF5_LIBS, HDF5_INCLUDE_DIRS,
            openmp=openmp, openmp_libs=OPENMP_LIBS
        )
        print(f"[{label}] 编译成功: {exe_path}" if success else f"[{label}] 编译失败")
        if success:
            compiled_refs[key] = {
                "exe": exe_path,
                "label": label,
                "opt_level": opt_level,
                "openmp": openmp,
                "runtime_env": runtime_env,
            }
    print("=" * 50)

    # ── 编译 EzComp ─────────────────────────────────────────────────────────────
    print("========== 编译 EzComp ==========")
    compiled_ezcomps = {}
    for cfg in ezcomp_configs:
        print(f"--- {cfg['label']} ---")
        success, exe_path = compile_ezcomp(
            EZCOMP_EXECUTABLE,
            comp_file,
            output_dir,
            args.test_dir,
            f"{args.test_dir}_{cfg['output_name']}",
            emit=cfg["emit"],
            pipeline=cfg.get("pipeline"),
        )
        if not success:
            print(f"[{cfg['label']}] 编译失败")
            return 1
        print(f"[{cfg['label']}] 编译成功: {exe_path}")
        compiled_ezcomps[cfg["key"]] = {
            "exe": exe_path,
            "label": cfg["label"],
            "runtime_env": cfg.get("runtime_env", ""),
        }
    print("=" * 50)

    # ── 运行性能测试 ───────────────────────────────────────────────────────────
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
                success, stderr_text = run_executable(
                    compiled_ezcomps[version_key]["exe"],
                    output_dir,
                    runtime_env=runtime_env,
                )
                if not success:
                    print(f"Round {run_idx + 1}: {label} 失败")
                    return 1
                run_time = extract_run_time(stderr_text)
            else:
                ref_info = compiled_refs[version_key]
                runtime_env = ref_info.get("runtime_env", "")
                success, stderr_text = run_executable(
                    ref_info["exe"],
                    output_dir,
                    [f"result_{ref_info['opt_level']}_{run_idx}.h5", f"{ref_info['opt_level']}_{run_idx}"],
                    runtime_env=runtime_env,
                )
                if not success:
                    print(f"Round {run_idx + 1}: {ref_info['label']} 失败")
                    return 1
                run_time = extract_run_time(stderr_text)

            run_times[version_key].append(run_time)

        print(f"Round {run_idx + 1}/{args.runs} 完成，本轮用时: {time.time() - round_start:.3f}s")
    print("=" * 50)

    # ── 统计结果 ───────────────────────────────────────────────────────────────
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

    # ── 验证结果 ───────────────────────────────────────────────────────────────
    print("========== 验证结果 ==========")
    all_data = {}
    if (ezcomp_data := load_h5_result(os.path.join(output_dir, "result.h5"))):
        all_data["ezcomp"] = ezcomp_data
        print(f"ezcomp result.h5: 数据集 {ezcomp_data['datasets']}")

    # 使用最后一个 reference 配置作为验证基准
    ref_config = reference_configs[-1]
    ref_key = ref_config["opt_level"]
    last_ref_file = f"result_{ref_key}_{args.runs - 1}.h5"
    if (ref_data := load_h5_result(os.path.join(output_dir, last_ref_file))):
        all_data[ref_key] = ref_data
        print(f"{ref_config['label']} {last_ref_file}: 数据集 {ref_data['datasets']}")

    passed, errors = verify_results(all_data, rtol=args.rtol)
    if not passed:
        print("\n❌ 结果验证失败!")
        for err in errors:
            print(f"  错误: {err}")
        return 1
    print("=" * 50)

    # ── 绘制性能比较图 ─────────────────────────────────────────────────────────
    if not args.no_plot:
        print("========== 绘制性能比较图 ==========")
        plot_performance(
            perf_data,
            output_path=os.path.join(output_dir, f"{args.test_dir}.pdf"),
            num_runs=args.runs,
        )
        print("=" * 50)

    # ── 清理中间文件 ───────────────────────────────────────────────────────────
    print("========== 清理中间文件 ==========")
    cleaned_count = cleanup_intermediates(output_dir)
    print(f"已清理 {cleaned_count} 个中间文件")
    print("=" * 50)

    print("\n✅ 所有测试通过!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
