"""Consistent report figures (single hue for one metric, fixed categorical order for several)."""
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order (validated palette)
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e6"


def _style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def method_bar(df, metric, label_col="method", title=None, ax=None, xlim=(0, 1)):
    """Horizontal bars, one metric per method, best on top, value at bar end."""
    d = df[[label_col, metric]].dropna().sort_values(metric)
    ax = ax or plt.subplots(figsize=(6, 0.45 * len(d) + 1))[1]
    ax.barh(d[label_col], d[metric], color=SERIES[0], height=0.6)
    for y, v in enumerate(d[metric]):
        ax.text(v + 0.01 * (xlim[1] - xlim[0]), y, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(*xlim)
    ax.set_title(title or metric, loc="left", color=INK, fontsize=11)
    _style(ax)
    plt.tight_layout()
    return ax


def grouped_bars(df, metrics, label_col="method", title=None, ax=None):
    """Several metrics (<=4) per method, legend always shown."""
    d = df[[label_col] + list(metrics)].dropna()
    n, k = len(d), len(metrics)
    ax = ax or plt.subplots(figsize=(max(6, 1.3 * n), 3.8))[1]
    x = np.arange(n)
    w = 0.8 / k
    for i, m in enumerate(metrics):
        ax.bar(x + (i - (k - 1) / 2) * w, d[m], width=w * 0.92, color=SERIES[i], label=m)
    ax.set_xticks(x, d[label_col], rotation=20, ha="right", color=INK)
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, ncol=k, loc="upper left", bbox_to_anchor=(0, 1.12))
    ax.set_title(title or "", loc="left", color=INK, fontsize=11, pad=24)
    plt.tight_layout()
    return ax
