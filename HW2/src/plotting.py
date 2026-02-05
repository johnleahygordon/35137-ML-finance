"""Plotting utilities for HW2."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sharpe_bar_chart(
    sharpes: pd.Series,
    title: str,
    save_path: str | Path,
    top_n: int = 5,
    figsize: tuple[int, int] = (14, 6),
) -> None:
    """Horizontal bar chart of Sharpe ratios with best/worst highlighted."""
    sharpes = sharpes.sort_values()
    colors = ["#999999"] * len(sharpes)
    # highlight best
    for i in range(min(top_n, len(sharpes))):
        colors[-(i + 1)] = "#2ca02c"
    # highlight worst
    for i in range(min(top_n, len(sharpes))):
        colors[i] = "#d62728"

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(sharpes)), sharpes.values, color=colors)
    ax.set_yticks(range(len(sharpes)))
    ax.set_yticklabels(sharpes.index, fontsize=5)
    ax.set_xlabel("Annualized Sharpe Ratio")
    ax.set_title(title)
    ax.axvline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compare_sharpe_bar(
    char_sharpes: pd.Series,
    ml_sharpes: pd.Series,
    title: str,
    save_path: str | Path,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Side-by-side bar chart comparing characteristic vs ML Sharpes."""
    all_labels = list(ml_sharpes.index)
    x = np.arange(len(all_labels))
    width = 0.35

    # get the best characteristic sharpe for reference
    best_char = char_sharpes.max()

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, ml_sharpes.values, width, label="ML Models", color="#1f77b4")
    ax.axhline(best_char, color="#2ca02c", linestyle="--",
               label=f"Best characteristic ({char_sharpes.idxmax()})")
    ax.axhline(char_sharpes.median(), color="#ff7f0e", linestyle=":",
               label="Median characteristic")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def explained_variance_plot(
    explained_var: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    """Cumulative explained variance with threshold lines."""
    cumvar = np.cumsum(explained_var)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", markersize=3)
    for thresh in [0.80, 0.90, 0.95]:
        ax.axhline(thresh, color="grey", linestyle="--", linewidth=0.7)
        n_above = int(np.searchsorted(cumvar, thresh) + 1)
        ax.text(len(cumvar) * 0.7, thresh + 0.005, f"{thresh:.0%}: {n_above} factors",
                fontsize=8)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def sharpe_vs_k_plot(
    k_values: list[int],
    sharpe_dict: dict[str, list[float]],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot OOS Sharpe vs number of PCA factors for multiple methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, sharpes in sharpe_dict.items():
        ax.plot(k_values[:len(sharpes)], sharpes, "o-", markersize=3, label=label)
    ax.set_xlabel("Number of PCA Factors (k)")
    ax.set_ylabel("Annualized Sharpe Ratio (OOS)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
