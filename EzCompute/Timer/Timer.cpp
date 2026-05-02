//===-- Timer.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// 运行时计时器
//
//===----------------------------------------------------------------------===//


#include <chrono>
#include <cstdio>
#include <cinttypes>

namespace ezcompute {

using Clock = std::chrono::steady_clock;

static thread_local Clock::time_point tls_start;

//===----------------------------------------------------------------------===//
// C ABI 导出函数
//===----------------------------------------------------------------------===//

/// 启动计时器
extern "C" void timer_start() {
	tls_start = Clock::now();
}

/// 停止计时并打印耗时到 stderr，格式为 [TIMER] Xh Xm Xs Xms
extern "C" void timer_stop_and_print() {
	auto end = Clock::now();
	auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - tls_start).count();

	auto hours = ns / 3600000000000LL;
	ns %= 3600000000000LL;
	auto minutes = ns / 60000000000LL;
	ns %= 60000000000LL;
	auto seconds = ns / 1000000000LL;
	ns %= 1000000000LL;
	auto milliseconds = ns / 1000000LL;

	std::fprintf(stderr, "[TIMER] %lldh %lldm %llds %lldms\n",
	             (long long)hours, (long long)minutes,
	             (long long)seconds, (long long)milliseconds);
}

}
