#===-- lit.cfg.py ---------------------------------------------*- Python -*-===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

import os
import lit.formats

# 配置名称
config.name = "EzComp"

# 测试文件后缀
config.suffixes = ['.comp', '.test']

# 测试格式
config.test_format = lit.formats.ShTest(True)

# 获取 EzComp 可执行文件路径
ezcomp_exe = lit_config.params.get('EZCOMP_EXECUTABLE', None)
if ezcomp_exe:
    config.substitutions.append(('%ezcomp', ezcomp_exe))
else:
    ezcomp_exe = os.environ.get('EZCOMP_EXECUTABLE', 'ezcomp')
    config.substitutions.append(('%ezcomp', ezcomp_exe))

# 添加 FileCheck
filecheck_exe = lit_config.params.get('FILECHECK_EXECUTABLE', None)
if filecheck_exe:
    config.substitutions.append(('FileCheck', filecheck_exe))

# 添加常用替换
config.substitutions.append(('%s', '%{input}'))