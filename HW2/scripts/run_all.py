#!/usr/bin/env python3
"""HW2 — run the full pipeline (Q1 + Q2) and write outputs."""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Objective did not converge.*")

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import load_lsret  # noqa: E402
from src.question1 import run_q1_for_dataset  # noqa: E402
from src.question2 import (  # noqa: E402
    _build_portfolio_returns_from_chars,
    run_q2a,
    run_q2b,
    run_q2c,
    run_q2e,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hw2")


def generate_report(out_dir: Path, q1_large: dict, q1_small: dict) -> None:
    """Write outputs/report.md summarising all results."""
    lines = [
        "# HW2 — ML in Finance: Results Report",
        "",
        "## Question 1",
        "",
        "### 1(a) Characteristic Portfolios — Large Caps",
        "",
        f"**{len(q1_large['char_sharpes'])} characteristics** evaluated.",
        "",
        f"- **Best**: {q1_large['char_sharpes'].idxmax()} "
        f"(Sharpe = {q1_large['char_sharpes'].max():.3f})",
        f"- **Worst**: {q1_large['char_sharpes'].idxmin()} "
        f"(Sharpe = {q1_large['char_sharpes'].min():.3f})",
        "",
        "See `q1a_large_sharpe_plot.png` and `q1a_large_sharpe_by_char.csv`.",
        "",
        "### 1(b) ML Models — Large Caps",
        "",
        "See `q1b_large_oos_r2_by_model.csv` and `q1b_large_hyperparams.json`.",
        "",
        "**Interpretation**: <!-- TODO: add interpretation -->",
        "",
        "### 1(c) ML Portfolio Sharpes — Large Caps",
        "",
    ]

    if q1_large["ml_sharpes"] is not None and len(q1_large["ml_sharpes"]) > 0:
        lines.append(f"- **Best ML**: {q1_large['ml_sharpes'].idxmax()} "
                     f"(Sharpe = {q1_large['ml_sharpes'].max():.3f})")

    lines += [
        "",
        "See `q1c_large_ml_sharpe_comparison.csv` and `q1c_large_ml_sharpe_plot.png`.",
        "",
        "### 1(d) Small Caps",
        "",
        f"**{len(q1_small['char_sharpes'])} characteristics** evaluated.",
        "",
        f"- **Best**: {q1_small['char_sharpes'].idxmax()} "
        f"(Sharpe = {q1_small['char_sharpes'].max():.3f})",
        f"- **Worst**: {q1_small['char_sharpes'].idxmin()} "
        f"(Sharpe = {q1_small['char_sharpes'].min():.3f})",
        "",
        "See corresponding `*_small_*` files.",
        "",
        "**Large vs Small comparison**: <!-- TODO: add comparison -->",
        "",
        "### 1(e) Max Sharpe Attempt",
        "",
        f"- Large caps: {q1_large['max_sharpe']:.3f}",
        f"- Small caps: {q1_small['max_sharpe']:.3f}",
        "",
        "**Strategy**: Ensemble of all ML model z-scored predictions, "
        "z-score weighted portfolio, volatility scaled to 15% annualized target.",
        "",
        "---",
        "",
        "## Question 2",
        "",
        "### 2(a) PCA Latent Factors",
        "",
        "See `q2a_explained_variance_*.png` and `q2a_factor_loadings_*.csv`.",
        "",
        "**Interpretation**: <!-- TODO: add interpretation -->",
        "",
        "### 2(b) Indicator Prediction (pre-2004)",
        "",
        "See `q2b_indicator_*.csv`.",
        "",
        "### 2(c) Sharpe vs # PCA Factors",
        "",
        "See `q2c_sharpe_vs_k_*.png` and `q2c_sharpe_vs_k_*.csv`.",
        "",
        "**Interpretation**: <!-- TODO: add interpretation -->",
        "",
        "### 2(d) Large/Small Comparison",
        "",
        "See `q2d_comparison.csv`.",
        "",
        "### 2(e) Max Sharpe Attempt (lsret)",
        "",
        "See `q2e_max_sharpe.csv`.",
        "",
        "---",
        "",
        "## Assumptions",
        "",
        "- Returns column (`ret`) contains string 'C' for some rows → treated as missing.",
        "- Missing characteristics imputed with cross-sectional median per month.",
        "- Features with >80% missing values excluded from ML models.",
        "- OOS R² computed against a zero-mean benchmark (i.e., SS_tot = sum(y²)).",
        "- Portfolio: equal-weight decile long-short (top minus bottom).",
        "- Time splits: first 20 unique years = train, next 12 = validation, rest = test.",
        "- Q2 indicator cutoff: data before 2004 for training.",
        "- Volatility scaling targets 15% annualized vol with leverage capped at [0.5, 3.0].",
        "",
    ]

    (out_dir / "report.md").write_text("\n".join(lines))
    logger.info("Report written to %s", out_dir / "report.md")


def main() -> None:
    t0 = time.time()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    np.random.seed(cfg.seed)

    out_dir = ROOT / cfg.output_dir
    out_dir.mkdir(exist_ok=True)

    # ── Q1: Large caps ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Q1 — LARGE CAPS")
    logger.info("=" * 60)
    q1_large = run_q1_for_dataset(str(ROOT / cfg.large_path), cfg, out_dir, label="large")

    # ── Q1: Small caps (Q1d) ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Q1 — SMALL CAPS (Q1d)")
    logger.info("=" * 60)
    q1_small = run_q1_for_dataset(str(ROOT / cfg.small_path), cfg, out_dir, label="small")

    # ── Q2 ────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Q2 — PCA & INDICATOR REPLICATION")
    logger.info("=" * 60)

    # Load lsret
    lsret = load_lsret(str(ROOT / cfg.lsret_path))
    lsret_date_col = "date"
    lsret_wide = lsret.set_index(lsret_date_col)

    # Build wide portfolio returns from Q1a characteristic portfolios
    large_wide = _build_portfolio_returns_from_chars(q1_large["df"], q1_large["schema"], cfg)
    small_wide = _build_portfolio_returns_from_chars(q1_small["df"], q1_small["schema"], cfg)

    # Q2(a)
    logger.info("── Q2(a) ──")
    q2a_results = run_q2a(lsret_wide, large_wide, small_wide, cfg, out_dir)

    # Q2(b) — lsret
    logger.info("── Q2(b) ──")
    q2b_lsret = run_q2b(
        lsret_wide, cfg.q2.indicator_cutoff_year,
        cfg.q2.ridge_alphas, cfg.q2.lasso_alphas,
        label="lsret", out_dir=out_dir,
    )

    # Q2(c) — lsret
    logger.info("── Q2(c) ──")
    q2c_lsret = run_q2c(
        lsret_wide, cfg.q2.indicator_cutoff_year,
        cfg.q2.ridge_alphas, cfg.q2.lasso_alphas,
        k_max=cfg.q2.pca_k_range_max,
        label="lsret", out_dir=out_dir,
    )

    # Q2(d) — repeat b+c for large and small
    logger.info("── Q2(d) ──")
    for lbl, wide in [("large_char", large_wide), ("small_char", small_wide)]:
        run_q2b(
            wide, cfg.q2.indicator_cutoff_year,
            cfg.q2.ridge_alphas, cfg.q2.lasso_alphas,
            label=lbl, out_dir=out_dir,
        )
        run_q2c(
            wide, cfg.q2.indicator_cutoff_year,
            cfg.q2.ridge_alphas, cfg.q2.lasso_alphas,
            k_max=cfg.q2.pca_k_range_max,
            label=lbl, out_dir=out_dir,
        )

    # Combine Q2d comparison
    q2d_parts = []
    for lbl in ["lsret", "large_char", "small_char"]:
        p_b = out_dir / f"q2b_indicator_{lbl}.csv"
        p_c = out_dir / f"q2c_sharpe_vs_k_{lbl}.csv"
        if p_b.exists():
            q2d_parts.append(pd.read_csv(p_b))
        if p_c.exists():
            q2d_parts.append(pd.read_csv(p_c))
    if q2d_parts:
        pd.concat(q2d_parts, ignore_index=True).to_csv(out_dir / "q2d_comparison.csv", index=False)

    # Q2(e)
    logger.info("── Q2(e) ──")
    run_q2e(
        lsret_wide, cfg.q2.indicator_cutoff_year,
        cfg.q2.ridge_alphas, cfg.q2.lasso_alphas,
        out_dir=out_dir,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    generate_report(out_dir, q1_large, q1_small)

    elapsed = time.time() - t0
    logger.info("Done in %.1f s. Outputs in %s", elapsed, out_dir)


if __name__ == "__main__":
    main()
