"""Question 1: Characteristic portfolios and ML signals."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import clean_panel, infer_schema, load_parquet, make_time_splits
from .ml_models import _safe_predict, fit_and_select
from .plotting import compare_sharpe_bar, sharpe_bar_chart
from .portfolio import (
    annualized_sharpe,
    long_short_portfolio_returns,
    oos_r2,
    rank_signal_by_month,
    zscore_weighted_portfolio_returns,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Q1(a): characteristic rank-sort portfolios
# ═══════════════════════════════════════════════════════════════════════════════

def run_q1a(
    df: pd.DataFrame,
    schema: dict,
    cfg: Config,
    out_dir: Path,
    prefix: str = "q1a",
) -> pd.Series:
    """Compute long-short portfolio Sharpe for each characteristic.

    Returns a Series: characteristic → annualized Sharpe.
    """
    date_col = schema["date"]
    ret_col = schema["ret"]
    features = schema["features"]

    sharpes: dict[str, float] = {}
    for feat in features:
        # Skip features with very high missingness (>80%)
        if df[feat].isna().mean() > 0.80:
            continue
        sig = rank_signal_by_month(df, feat, date_col=date_col)
        ls_ret = long_short_portfolio_returns(
            df, sig, ret_col=ret_col, date_col=date_col,
            n_quantiles=cfg.portfolio.n_quantiles,
        )
        if len(ls_ret) < 24:
            continue
        sharpes[feat] = annualized_sharpe(ls_ret)

    sharpe_series = pd.Series(sharpes).sort_values(ascending=False)
    sharpe_series.to_csv(out_dir / f"{prefix}_sharpe_by_char.csv")

    sharpe_bar_chart(
        sharpe_series,
        title=f"{prefix.upper()}: Annualized Sharpe by Characteristic",
        save_path=out_dir / f"{prefix}_sharpe_plot.png",
    )

    logger.info("Q1a (%s): %d characteristics evaluated", prefix, len(sharpe_series))
    logger.info("  Best:  %s (%.3f)", sharpe_series.idxmax(), sharpe_series.max())
    logger.info("  Worst: %s (%.3f)", sharpe_series.idxmin(), sharpe_series.min())
    return sharpe_series


# ═══════════════════════════════════════════════════════════════════════════════
# Q1(b): ML signals
# ═══════════════════════════════════════════════════════════════════════════════

def run_q1b(
    df: pd.DataFrame,
    schema: dict,
    cfg: Config,
    out_dir: Path,
    prefix: str = "q1b",
) -> tuple[dict, pd.DataFrame]:
    """Train ML models, select on validation, evaluate OOS R² on test.

    Returns (results_dict, predictions_df).
    """
    date_col = schema["date"]
    ret_col = schema["ret"]
    features = schema["features"]

    # Keep only features that are usable (not too sparse)
    usable = [f for f in features if df[f].isna().mean() < 0.80]
    logger.info("Q1b: using %d / %d features (dropping >80%% missing)", len(usable), len(features))

    # Fill remaining NaN with 0 for ML (after median imputation in clean_panel)
    X_all = df[usable].fillna(0).values
    y_all = df[ret_col].values.astype(float)
    dates = df[date_col]

    train_mask, val_mask, test_mask = make_time_splits(
        dates, train_years=cfg.split.train_years, val_years=cfg.split.val_years
    )

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    logger.info("Train: %d  Val: %d  Test: %d", len(y_train), len(y_val), len(y_test))

    # Fit all models
    results = fit_and_select(X_train, y_train, X_val, y_val, cfg)

    # Evaluate on test
    oos_r2_dict: dict[str, float] = {}
    all_preds = {}
    for name, res in results.items():
        model = res["model"]
        if model is None:
            continue
        pred = _safe_predict(model, X_test)
        r2 = oos_r2(y_test, pred)
        oos_r2_dict[name] = r2
        all_preds[name] = pred

    oos_df = pd.DataFrame([
        {"model": k, "oos_r2": v} for k, v in oos_r2_dict.items()
    ]).sort_values("oos_r2", ascending=False)
    oos_df.to_csv(out_dir / f"{prefix}_oos_r2_by_model.csv", index=False)

    # Hyperparameters
    hp = {name: res["params"] for name, res in results.items()}
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    hp_clean = {k: {kk: _convert(vv) for kk, vv in v.items()} for k, v in hp.items()}
    with open(out_dir / f"{prefix}_hyperparams.json", "w") as f:
        json.dump(hp_clean, f, indent=2)

    # Build predictions dataframe (test period only)
    pred_df = df.loc[test_mask, [schema["id"], date_col, ret_col]].copy()
    for name, pred in all_preds.items():
        pred_df[f"pred_{name}"] = pred
    pred_df.to_parquet(out_dir / f"{prefix}_predictions.parquet", index=False)

    logger.info("Q1b (%s) OOS R²:", prefix)
    for _, row in oos_df.iterrows():
        logger.info("  %-20s  %.6f", row["model"], row["oos_r2"])

    return results, pred_df


# ═══════════════════════════════════════════════════════════════════════════════
# Q1(c): ML portfolio Sharpes
# ═══════════════════════════════════════════════════════════════════════════════

def run_q1c(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    schema: dict,
    cfg: Config,
    out_dir: Path,
    prefix: str = "q1c",
) -> pd.Series:
    """Form portfolios from ML predictions and compare Sharpes.

    Characteristic Sharpes are recomputed on the test period only so the
    comparison with ML portfolios covers the same time window.
    """
    date_col = schema["date"]
    ret_col = schema["ret"]
    features = schema["features"]

    # ── Recompute characteristic Sharpes on the test period only ──────────
    test_dates = pred_df[date_col].unique()
    df_test = df[df[date_col].isin(test_dates)]

    char_sharpes: dict[str, float] = {}
    for feat in features:
        if df_test[feat].isna().mean() > 0.80:
            continue
        sig = rank_signal_by_month(df_test, feat, date_col=date_col)
        ls_ret = long_short_portfolio_returns(
            df_test, sig, ret_col=ret_col, date_col=date_col,
            n_quantiles=cfg.portfolio.n_quantiles,
        )
        if len(ls_ret) < 12:
            continue
        char_sharpes[feat] = annualized_sharpe(ls_ret)

    char_sharpe_series = pd.Series(char_sharpes).sort_values(ascending=False)

    # ── ML portfolio Sharpes ─────────────────────────────────────────────
    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    ml_sharpes: dict[str, float] = {}

    for col in pred_cols:
        model_name = col.replace("pred_", "")
        ls_ret = long_short_portfolio_returns(
            pred_df, col, ret_col=ret_col, date_col=date_col,
            n_quantiles=cfg.portfolio.n_quantiles,
        )
        if len(ls_ret) < 12:
            continue
        ml_sharpes[model_name] = annualized_sharpe(ls_ret)

    ml_sharpe_series = pd.Series(ml_sharpes).sort_values(ascending=False)

    # ── Save comparison table ────────────────────────────────────────────
    comparison = pd.DataFrame({
        "model": list(ml_sharpe_series.index),
        "sharpe": list(ml_sharpe_series.values),
        "type": "ML",
    })
    top_chars = char_sharpe_series.nlargest(5)
    bottom_chars = char_sharpe_series.nsmallest(5)
    char_rows = pd.DataFrame({
        "model": list(top_chars.index) + list(bottom_chars.index),
        "sharpe": list(top_chars.values) + list(bottom_chars.values),
        "type": "Characteristic",
    })
    full_comparison = pd.concat([comparison, char_rows], ignore_index=True)
    full_comparison.to_csv(out_dir / f"{prefix}_ml_sharpe_comparison.csv", index=False)

    # Plot
    compare_sharpe_bar(
        char_sharpe_series, ml_sharpe_series,
        title=f"{prefix.upper()}: ML vs Characteristic Sharpes",
        save_path=out_dir / f"{prefix}_ml_sharpe_plot.png",
    )

    logger.info("Q1c (%s) ML Sharpes:", prefix)
    for name, s in ml_sharpe_series.items():
        logger.info("  %-20s  %.3f", name, s)

    return ml_sharpe_series


# ═══════════════════════════════════════════════════════════════════════════════
# Q1(e): Max Sharpe attempt
# ═══════════════════════════════════════════════════════════════════════════════

def run_q1e(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    schema: dict,
    cfg: Config,
    out_dir: Path,
    prefix: str = "q1e",
) -> float:
    """Attempt to maximize OOS Sharpe by combining signals."""
    date_col = schema["date"]
    ret_col = schema["ret"]

    pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]
    if not pred_cols:
        logger.warning("No predictions available for Q1e")
        return np.nan

    # Strategy: z-score weighted ensemble of all ML predictions
    # plus volatility scaling
    pred_df = pred_df.copy()

    # Ensemble: average z-score of all model predictions per month
    def _ensemble_signal(g):
        preds = g[pred_cols]
        zscores = []
        for col in pred_cols:
            s = preds[col]
            std = s.std()
            if std > 0:
                zscores.append((s - s.mean()) / std)
        if not zscores:
            return pd.Series(0.0, index=g.index)
        z_df = pd.concat(zscores, axis=1)
        return z_df.mean(axis=1)

    pred_df["ensemble_signal"] = pred_df.groupby(date_col, group_keys=False).apply(_ensemble_signal)

    # Form long-short with z-score weighting (more aggressive)
    ls_ret = zscore_weighted_portfolio_returns(
        pred_df, "ensemble_signal", ret_col=ret_col, date_col=date_col,
    )

    # Volatility scaling: target 15% annualized vol
    target_vol = 0.15
    rolling_vol = ls_ret.rolling(12, min_periods=6).std() * np.sqrt(12)
    scale = target_vol / rolling_vol.shift(1)
    scale = scale.clip(0.5, 3.0)  # bound leverage
    scaled_ret = ls_ret * scale
    scaled_ret = scaled_ret.dropna()

    sr = annualized_sharpe(scaled_ret)

    # Save
    result = pd.DataFrame({
        "strategy": ["Ensemble + VolScaling"],
        "sharpe": [sr],
        "description": [
            "Average z-scored predictions across all ML models, "
            "z-score weighted portfolio, volatility scaled to 15% annualized"
        ],
    })
    result.to_csv(out_dir / f"{prefix}_max_sharpe.csv", index=False)

    logger.info("Q1e (%s): Max Sharpe attempt = %.3f", prefix, sr)
    return sr


# ═══════════════════════════════════════════════════════════════════════════════
# Full Q1 pipeline for a single dataset
# ═══════════════════════════════════════════════════════════════════════════════

def run_q1_for_dataset(
    path: str,
    cfg: Config,
    out_dir: Path,
    label: str,  # "large" or "small"
) -> dict:
    """Run Q1(a)–(c)+(e) for a single dataset. Returns summary dict."""
    prefix_a = f"q1a_{label}"
    prefix_b = f"q1b_{label}"
    prefix_c = f"q1c_{label}"
    prefix_e = f"q1e_{label}"

    # Load and clean
    df = load_parquet(path)
    schema = infer_schema(df, id_col=cfg.id_col, date_col=cfg.date_col, ret_col=cfg.ret_col)
    df = clean_panel(df, cfg.id_col, cfg.date_col, cfg.ret_col, schema["features"])
    # Re-infer after cleaning (row count changes)
    schema = infer_schema(df, id_col=cfg.id_col, date_col=cfg.date_col, ret_col=cfg.ret_col)

    # Q1(a)
    char_sharpes = run_q1a(df, schema, cfg, out_dir, prefix=prefix_a)

    # Q1(b)
    results, pred_df = run_q1b(df, schema, cfg, out_dir, prefix=prefix_b)

    # Q1(c)
    ml_sharpes = run_q1c(df, pred_df, schema, cfg, out_dir, prefix=prefix_c)

    # Q1(e)
    max_sr = run_q1e(df, pred_df, schema, cfg, out_dir, prefix=prefix_e)

    return {
        "label": label,
        "char_sharpes": char_sharpes,
        "ml_sharpes": ml_sharpes,
        "pred_df": pred_df,
        "max_sharpe": max_sr,
        "schema": schema,
        "df": df,
    }
