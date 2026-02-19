"""Evaluation metrics, panel assembly, robustness checks, and P&L strategy.

Primary evaluation: LOO-CV RMSE, MAE, and directional accuracy.
Also computes OOS R² vs predict-zero benchmark.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models import (
    FEATURE_SETS,
    build_elasticnet,
    build_gbr,
    build_historical_mean,
    build_huber,
    build_lasso,
    build_ols,
    build_predict_zero,
    build_ridge,
    get_feature_cols,
    lomo_cv,
)

logger = logging.getLogger(__name__)


# ── Panel assembly ────────────────────────────────────────────────────────────

# Maps target window → text_source for joining text features
_WINDOW_TO_TEXT_SOURCE = {
    "statement": "statement",
    "digestion": "press_conf",
}


def assemble_panel(
    targets: pd.DataFrame,
    feat_structured: pd.DataFrame,
    feat_text: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge targets + structured features + text features into the modeling panel.

    Text features are joined on (meeting_id, text_source) where:
        statement window ← text_source == 'statement'
        digestion window ← text_source == 'press_conf'

    Returns the final panel ready for modeling.
    """
    # Start with valid targets only
    panel = targets[targets["has_data"]].copy()

    # Join structured features (keyed by meeting_id × pair)
    panel = panel.merge(feat_structured, on=["meeting_id", "pair"], how="left")

    if feat_text is not None:
        # Add text_source column to panel based on window
        panel["text_source"] = panel["window"].map(_WINDOW_TO_TEXT_SOURCE)
        panel = panel.merge(
            feat_text, on=["meeting_id", "text_source"], how="left"
        )

    logger.info("Panel assembled: %d rows, %d columns", len(panel), panel.shape[1])
    return panel.reset_index(drop=True)


# ── Evaluation metrics ────────────────────────────────────────────────────────


def compute_metrics(actuals: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, directional accuracy, and OOS R²."""
    valid = ~(np.isnan(actuals) | np.isnan(preds))
    a = actuals[valid]
    p = preds[valid]

    if len(a) == 0:
        return {k: np.nan for k in ["mae", "rmse", "dir_acc", "oos_r2", "n"]}

    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    dir_acc = float(np.mean(np.sign(a) == np.sign(p)))
    # OOS R² vs predict-zero: 1 - SS_res / SS_tot  (SS_tot = sum(a^2) for zero baseline)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum(a ** 2)
    oos_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc, "oos_r2": oos_r2, "n": int(valid.sum())}


# ── Full ladder run ───────────────────────────────────────────────────────────


def run_ladder(
    panel: pd.DataFrame,
    cfg,
    target_col: str = "log_ret",
    pairs: list[str] | None = None,
    windows: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full model ladder (rungs 1–4) and return a results DataFrame.

    For each (rung, window, pair) combination, runs LOO-CV and records metrics.
    Also runs a pooled (all-pairs) model for each rung and window.

    Returns a DataFrame with columns:
        rung, model_name, window, pair, mae, rmse, dir_acc, oos_r2, n
    """
    if pairs is None:
        pairs = panel["pair"].unique().tolist()
    if windows is None:
        windows = panel["window"].unique().tolist()

    emb_cols = [c for c in panel.columns if c.startswith("emb_pc_")]

    # Define rung configurations
    rung_configs = [
        # (rung_label, model_name, model_builder, feature_set_names)
        ("Rung 1 – Predict Zero",    "predict_zero",    build_predict_zero,    []),
        ("Rung 1 – Hist Mean",       "hist_mean",       build_historical_mean, []),
        ("Rung 2 – OLS Theory",      "ols_theory",      build_ols,             ["rung2_theory"]),
        ("Rung 3a – Ridge",          "ridge",           lambda: build_ridge(1.0),       ["rung3_structured"]),
        ("Rung 3a – LASSO",          "lasso",           lambda: build_lasso(0.01),      ["rung3_structured"]),
        ("Rung 3a – ElasticNet",     "elasticnet",      lambda: build_elasticnet(0.01), ["rung3_structured"]),
        ("Rung 3a – Huber",          "huber",           build_huber,           ["rung3_structured"]),
        ("Rung 3b – GBR",            "gbr",             build_gbr,             ["rung3_structured"]),
        ("Rung 4 – +Keywords",       "rung4_kw",        build_ridge,
            ["rung3_structured", "rung4_text_kw"]),
        ("Rung 4 – +LLM Rubric",     "rung4_llm",       build_ridge,
            ["rung3_structured", "rung4_text_kw", "rung4_text_llm"]),
        ("Rung 4 – +Embeddings",     "rung4_emb",       build_ridge,
            ["rung3_structured", "rung4_text_kw", "rung4_text_llm", "rung4_embeddings"]),
    ]

    results: list[dict] = []

    for window in windows:
        logger.info("=== Window: %s ===", window)
        for rung_label, model_name, model_builder, feat_set_names in rung_configs:
            feature_cols = get_feature_cols(panel, feat_set_names)

            # Per-pair models
            for pair in pairs:
                sub = panel[(panel["window"] == window) & (panel["pair"] == pair)].copy()
                if len(sub) < 5:
                    continue

                if not feature_cols:
                    # Rung 1: no features needed
                    result = lomo_cv(sub, model_builder, ["is_hike"] if "is_hike" in sub.columns else [feature_cols[0]] if feature_cols else [], target_col)
                    # Special case: just call predict directly
                    mdl = model_builder()
                    valid = sub[target_col].notna()
                    actuals = sub.loc[valid, target_col].values
                    try:
                        mdl.fit(np.zeros((len(actuals), 1)), actuals)
                        preds = mdl.predict(np.zeros((valid.sum(), 1)))
                    except Exception:
                        preds = np.zeros(valid.sum())
                    metrics = compute_metrics(actuals, preds)
                else:
                    result = lomo_cv(sub, model_builder, feature_cols, target_col)
                    metrics = compute_metrics(result["actuals"].values, result["preds"].values)

                results.append(
                    {
                        "rung": rung_label,
                        "model_name": model_name,
                        "window": window,
                        "pair": pair,
                        **metrics,
                    }
                )

            # Pooled (all pairs) model
            sub_all = panel[panel["window"] == window].copy()
            if len(sub_all) >= 10 and feature_cols:
                result_pooled = lomo_cv(sub_all, model_builder, feature_cols, target_col)
                metrics_pooled = compute_metrics(result_pooled["actuals"].values, result_pooled["preds"].values)
            else:
                metrics_pooled = {k: np.nan for k in ["mae", "rmse", "dir_acc", "oos_r2", "n"]}

            results.append(
                {
                    "rung": rung_label,
                    "model_name": model_name,
                    "window": window,
                    "pair": "ALL (pooled)",
                    **metrics_pooled,
                }
            )

            logger.info("  %s | %s | pooled RMSE=%.5f dir_acc=%.3f", rung_label, window, metrics_pooled.get("rmse", np.nan), metrics_pooled.get("dir_acc", np.nan))

    return pd.DataFrame(results)


# ── Robustness checks ─────────────────────────────────────────────────────────


def robustness_outlier_drop(
    panel: pd.DataFrame,
    n_drop: int = 3,
    target_col: str = "log_ret",
) -> pd.DataFrame:
    """Drop the n_drop largest absolute-return meetings and re-run the ladder.

    Used to check whether results are driven by a few extreme events.
    """
    abs_by_meeting = (
        panel.groupby("meeting_id")[target_col]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
    )
    drop_meetings = abs_by_meeting.iloc[:n_drop].index.tolist()
    logger.info("Outlier robustness: dropping meetings %s", drop_meetings)
    return panel[~panel["meeting_id"].isin(drop_meetings)].copy()


def robustness_direction_accuracy(
    panel: pd.DataFrame,
    model_builder,
    feature_cols: list[str],
    target_col: str = "log_ret",
) -> pd.DataFrame:
    """Classification version: predict direction (sign) rather than magnitude."""
    panel = panel.copy()
    panel["direction_target"] = np.sign(panel[target_col])
    result = lomo_cv(panel, model_builder, feature_cols, "direction_target")
    return pd.DataFrame(
        {
            "actual_direction": result["actuals"],
            "pred_direction": np.sign(result["preds"]),
            "meeting_id": result["meeting_ids"],
        }
    )


# ── P&L strategy check ───────────────────────────────────────────────────────


def strategy_pnl(
    panel: pd.DataFrame,
    model_builder,
    feature_cols: list[str],
    target_col: str = "log_ret",
    transaction_cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Simple long/short strategy based on model signal.

    Position: +1 if predicted return > 0, −1 if < 0.
    P&L: position × actual_return − |transaction_cost|.

    Returns per-row P&L DataFrame with cumulative P&L column.
    """
    result = lomo_cv(panel, model_builder, feature_cols, target_col)

    pnl_df = pd.DataFrame(
        {
            "meeting_id": result["meeting_ids"],
            "pair": result["pairs"],
            "window": result["windows"],
            "actual": result["actuals"],
            "pred": result["preds"],
        }
    )
    pnl_df = pnl_df.dropna(subset=["actual", "pred"])

    tc_bps = transaction_cost_bps / 10_000
    pnl_df["position"] = np.sign(pnl_df["pred"])
    pnl_df["gross_ret"] = pnl_df["position"] * pnl_df["actual"]
    pnl_df["net_ret"] = pnl_df["gross_ret"] - tc_bps  # flat cost per trade
    pnl_df["cum_gross"] = pnl_df["gross_ret"].cumsum()
    pnl_df["cum_net"] = pnl_df["net_ret"].cumsum()

    n = len(pnl_df)
    gross_sharpe = (pnl_df["gross_ret"].mean() / pnl_df["gross_ret"].std() * np.sqrt(n)) if pnl_df["gross_ret"].std() > 0 else np.nan
    net_sharpe = (pnl_df["net_ret"].mean() / pnl_df["net_ret"].std() * np.sqrt(n)) if pnl_df["net_ret"].std() > 0 else np.nan

    logger.info(
        "Strategy P&L: gross Sharpe=%.2f  net Sharpe=%.2f  (n=%d, tc=%.0f bps)",
        gross_sharpe, net_sharpe, n, transaction_cost_bps,
    )

    pnl_df.attrs["gross_sharpe"] = gross_sharpe
    pnl_df.attrs["net_sharpe"] = net_sharpe
    return pnl_df
