#===-- plotter.py --------------------------------------------*- Python -*-===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

"""
绘图模块
- 性能对比柱状图
- 支持学术风格输出（PDF 矢量图 + PNG 预览）
"""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np


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


# ─────────────────────────────────────────────────────────────────────────────
#  字体配置
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
#  性能对比柱状图
# ─────────────────────────────────────────────────────────────────────────────


def plot_performance(
    perf_data,
    output_path="performance_comparison.pdf",
    num_runs=10,
):
    """
    绘制学术风格性能对比柱状图（Tableau 10 配色）

    Args:
        perf_data: 性能数据字典，包含:
            - labels: 标签列表
            - times: 平均时间列表
            - stds: 标准差列表
            - n_ref: reference 数量
            - n_ez: ezcomp 数量
        output_path: 输出文件路径（PDF）
        num_runs: 运行次数（用于标注）

    输出:
        - PDF 矢量图
        - PNG 预览图 (300 dpi)
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
    fig_w_inch = max(5.0, n_bars * 0.9 + 1.0)
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
    ax.set_ylabel(
        "计算时间 (s)",
        fontproperties=_get_cn_font(size=11),
        labelpad=8,
    )
    # 调整 y 轴范围以容纳误差棒
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
