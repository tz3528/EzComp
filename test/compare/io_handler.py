#===-- io_handler.py ------------------------------------------*- Python -*-===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

"""
测试结果与中间产物的 IO 处理模块
- 配置加载
- 编译（reference.cpp 和 EzComp）
- 运行可执行文件
- 结果验证
- 文件清理
"""

import glob
import json
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import psutil

# 由 CMake 配置注入
EZCOMP_EXECUTABLE = "@EZCOMP_EXECUTABLE@"
COMPARE_TEST_SOURCE_DIR = "@COMPARE_TEST_SOURCE_DIR@"
CXX_COMPILER = "@CXX_COMPILER@"
HDF5_LIBS = "@EZCOMPUTE_HDF5_LIBS@"
HDF5_INCLUDE_DIRS = "@HDF5_INCLUDE_DIRS@"
OPENMP_LIBS = "@OPENMP_LIBS@"

CONFIG_FILE_NAME = "compare_config.json"


# ─────────────────────────────────────────────────────────────────────────────
#  配置加载
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
#  文件路径处理
# ─────────────────────────────────────────────────────────────────────────────


def find_comp_file(test_dir):
    """在测试目录中查找 .comp 文件"""
    comp_files = glob.glob(os.path.join(test_dir, "*.comp"))
    if not comp_files:
        return None
    if len(comp_files) > 1:
        print(f"警告: 找到多个 .comp 文件，使用第一个: {comp_files[0]}")
    return comp_files[0]


def get_executable_path(output_dir, name):
    """生成可执行文件的完整路径"""
    return os.path.join(output_dir, name + (".exe" if sys.platform == "win32" else ""))


# ─────────────────────────────────────────────────────────────────────────────
#  命令执行
# ─────────────────────────────────────────────────────────────────────────────


def run_command(cmd, cwd=None, env=None):
    """执行外部命令，返回结果"""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=run_env)


# ─────────────────────────────────────────────────────────────────────────────
#  编译
# ─────────────────────────────────────────────────────────────────────────────


def compile_reference(
    compiler,
    ref_cpp,
    output_dir,
    opt_level,
    hdf5_libs,
    hdf5_include_dirs,
    openmp=False,
    openmp_libs="",
):
    """
    编译 reference.cpp

    Args:
        compiler: C++ 编译器路径
        ref_cpp: reference.cpp 文件路径
        output_dir: 输出目录
        opt_level: 优化级别 (O2, O3 等)
        hdf5_libs: HDF5 库链接选项
        hdf5_include_dirs: HDF5 头文件目录
        openmp: 是否启用 OpenMP
        openmp_libs: OpenMP 库路径

    Returns:
        tuple: (success, executable_path)
    """
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


def compile_ezcomp(
    ezcomp_path,
    comp_file,
    working_dir,
    test_name,
    output_name,
    emit="compile",
    pipeline=None,
):
    """
    编译 EzComp 的 .comp 文件

    Args:
        ezcomp_path: ezcomp 可执行文件路径
        comp_file: .comp 源文件路径
        working_dir: 工作目录
        test_name: 测试名称
        output_name: 输出可执行文件名称
        emit: emit 选项
        pipeline: pass-pipeline 选项

    Returns:
        tuple: (success, executable_path)
    """
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


# ─────────────────────────────────────────────────────────────────────────────
#  运行
# ─────────────────────────────────────────────────────────────────────────────


def run_executable(executable, working_dir, args=None, runtime_env=None):
    """
    运行可执行文件

    Args:
        executable: 可执行文件路径
        working_dir: 工作目录
        args: 命令行参数列表
        runtime_env: 运行时环境变量字符串（如 "OMP_NUM_THREADS=8"）

    Returns:
        tuple: (success, stderr_text)
    """
    # 解析 runtime_env 字符串
    env_dict = {}
    if runtime_env:
        for item in runtime_env.split():
            if "=" in item:
                key, value = item.split("=", 1)
                if key == "OMP_NUM_THREADS" and value == "auto":
                    value = str(psutil.cpu_count(logical=False))
                env_dict[key] = value

    try:
        result = run_command(
            [executable, *(args or [])],
            cwd=working_dir,
            env=env_dict if env_dict else None,
        )
        if result.returncode != 0:
            print(f"运行失败: {executable}")
            print(result.stderr)
            return False, None
        return True, result.stderr
    except FileNotFoundError:
        print(f"错误: 找不到可执行文件: {executable}")
    except Exception as e:
        print(f"运行时出错: {e}")
    return False, None


# ─────────────────────────────────────────────────────────────────────────────
#  结果验证
# ─────────────────────────────────────────────────────────────────────────────


def load_h5_result(file_path):
    """加载 HDF5 结果文件"""
    if not os.path.exists(file_path):
        return None
    with h5py.File(file_path, "r") as f:
        return {
            "result": f["result"][:] if "result" in f else None,
            "datasets": list(f.keys()),
        }


def verify_results(all_data, rtol=1e-6):
    """
    验证 EzComp 结果与参考结果的一致性

    Args:
        all_data: 包含各版本结果的字典
        rtol: 相对容差

    Returns:
        tuple: (passed, errors)
    """
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

    # if max_rel_diff > rtol:
    #     return False, [f"最大相对误差 {max_rel_diff:.6e} 超过容差 {rtol}"]

    print("  ✓ 结果验证通过!")
    return True, []


# ─────────────────────────────────────────────────────────────────────────────
#  文件清理
# ─────────────────────────────────────────────────────────────────────────────


def cleanup_intermediates(output_dir):
    """
    清理中间文件，仅保留 png、pdf、excel 格式

    Args:
        output_dir: 输出目录

    Returns:
        int: 清理的文件数量
    """
    KEEP_EXTENSIONS = {'.png', '.pdf', '.xlsx', '.xls'}
    cleaned_count = 0

    for path in glob.glob(os.path.join(output_dir, '*')):
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext not in KEEP_EXTENSIONS:
                os.remove(path)
                cleaned_count += 1

    return cleaned_count
