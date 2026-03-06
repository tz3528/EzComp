//===-- LinkerOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EZ_COMPILE_LINKER_OPTIONS_H
#define EZ_COMPILE_LINKER_OPTIONS_H

#include <string>
#include <vector>

namespace ezcompile::link {

struct LinkerConfig {
    std::string objectFile;
    std::string outputFile = "a.out";
    std::vector<std::string> libraries;
    std::vector<std::string> archives;
    bool verbose = false;
};

} // namespace ezcompile::link

#endif // EZ_COMPILE_LINKER_OPTIONS_H