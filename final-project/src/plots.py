"""Reusable plotting utilities for the FX @ FOMC project.

All functions accept a matplotlib Axes object (or create their own figure)
and return the figure for saving. Designed to be called from both notebooks
and run_pipeline.py.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ── Colour palette ────────────────────────────────────────────────────────────

RUNG_COLORS = {
    "Rung 1": "#aec7e8",
    "Rung 2": "#ffbb78",
    "Rung 3": "#98df8a",
    "Rung 4": "#ff9896",
}

WINDOW_COLORS = {
    "statement": "steelblue",
    "digestion": "darkorange",
}

PAIR_COLORS = {
    "USDEUR": "#1f77b4",
    "USDJPY": "#ff7f0e",
    "USDGBP": "#2ca02c",
    "USDCAD": "#9467bd",
}


def _rung_color(rung_label: str) -> str:
    for k, v in RUNG_COLORS.items():
        if k in rung_label:
            return v
    return "#c7c7c7"


# ── Ladder chart ──────────────────────────────────────────────────────────────


def ladder_chart(
    results: pd.DataFrame,
    metric: str = "rmse",
    window: str = "statement",
    pair: str = "ALL (pooled)",
    rung_order: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart showing model performance by rung.

    metric: 'rmse', 'mae', or 'dir_acc'
    """
    sub = results[(results["window"] == window) & (results["pair"] == pair)].copy()

    if rung_order is None:
        rung_order = sub["rung"].tolist()
    sub = sub.set_index("rung").reindex(rung_order).reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    values = sub[metric].values * (10_000 if metric in ("rmse", "mae") else 100)
    colors = [_rung_color(r) for r in sub["rung"]]

    bars = ax.barh(range(len(sub)), values, color=colors, edgecolor="white", height=0.7)

    if metric == "dir_acc":
        ax.axvline(50, color="red", lw=1.5, ls="--", label="50% (random)")
        ax.legend(fontsize=9)

    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([r.replace(" – ", "\n") for r in sub["rung"]], fontsize=9)

    unit = "bps" if metric in ("rmse", "mae") else "%"
    ax.set_xlabel(f"{metric.upper()} ({unit})")

    direction = "(lower = better)" if metric != "dir_acc" else "(higher = better)"
    ax.set_title(title or f"{metric.upper()} — {window} window {direction}", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center", ha="left", fontsize=8,
            )

    # Legend for rung colours
    legend_elements = [
        mpatches.Patch(facecolor=v, label=k) for k, v in RUNG_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right" if metric != "dir_acc" else "lower left", fontsize=8)

    return fig


# ── Coverage heatmap ──────────────────────────────────────────────────────────


def coverage_heatmap(
    bars: pd.DataFrame,
    fomc_meta: pd.DataFrame,
    expected_bars: dict[str, int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Heatmap of intraday bar coverage (meeting × source)."""
    from src.clean import coverage_report

    cov = coverage_report(bars, fomc_meta)
    source_cols = [c for c in cov.columns if c != "announcement_et"]

    if expected_bars is None:
        expected_bars = {s: 288 if s not in ("SPX", "VIX") else 78 for s in source_cols}

    cov_pct = cov[source_cols].copy().astype(float)
    for s in source_cols:
        cov_pct[s] = (cov[s] / expected_bars.get(s, 288)).clip(0, 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        cov_pct, ax=ax,
        cmap="YlGn", vmin=0, vmax=1,
        linewidths=0.3,
        annot=cov[source_cols].astype(int),
        fmt="d", annot_kws={"size": 7},
        cbar_kws={"label": "Coverage (fraction of expected bars)"},
    )
    ax.set_title("Intraday Bar Coverage")
    ax.set_xlabel("Source")
    ax.set_ylabel("Meeting ID")
    return fig


# ── Feature-return scatter ────────────────────────────────────────────────────


def feature_return_scatter(
    panel: pd.DataFrame,
    feature: str,
    pair: str = "USDEUR",
    window: str = "statement",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter of one feature vs FX log-return with trend line and correlation."""
    sub = panel[
        (panel["pair"] == pair) & (panel["window"] == window)
    ].dropna(subset=[feature, "log_ret"]).copy()

    sub["log_ret_bps"] = sub["log_ret"] * 10_000

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    ax.scatter(sub[feature], sub["log_ret_bps"], alpha=0.7, s=40)

    if len(sub) > 2:
        z = np.polyfit(sub[feature], sub["log_ret_bps"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sub[feature].min(), sub[feature].max(), 50)
        ax.plot(x_line, p(x_line), "r--", lw=1.5)
        r = sub[[feature, "log_ret_bps"]].corr().iloc[0, 1]
        ax.set_title(f"{pair} {window}\n{feature}  r={r:.2f}")
    else:
        ax.set_title(f"{pair} {window}\n{feature}")

    ax.set_xlabel(feature)
    ax.set_ylabel("FX log-return (bps)")
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.grid(True, alpha=0.3)
    return fig


# ── P&L chart ────────────────────────────────────────────────────────────────


def pnl_chart(
    pnl_df: pd.DataFrame,
    transaction_cost_bps: float = 2.0,
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Figure:
    """Cumulative gross and net P&L chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()

    ax.plot(range(len(pnl_df)), pnl_df["cum_gross"] * 10_000, lw=1.5, label="Gross")
    ax.plot(range(len(pnl_df)), pnl_df["cum_net"] * 10_000, lw=1.5, ls="--",
            label=f"Net (tc={transaction_cost_bps:.0f}bps)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(title or "Cumulative P&L")
    ax.set_xlabel("Trade index")
    ax.set_ylabel("Cumulative return (bps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# ── Save helper ───────────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, path: pathlib.Path, dpi: int = 150) -> None:
    """Save figure and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
