"""Portfolio construction and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── signal construction ───────────────────────────────────────────────────────

def rank_signal_by_month(
    df: pd.DataFrame,
    feature: str,
    date_col: str = "yyyymm",
) -> pd.Series:
    """Cross-sectional percentile rank (0–1) of *feature* within each month."""
    return df.groupby(date_col)[feature].rank(pct=True)


# ── portfolio returns ─────────────────────────────────────────────────────────

def long_short_portfolio_returns(
    df: pd.DataFrame,
    signal: str | pd.Series,
    ret_col: str = "ret",
    date_col: str = "yyyymm",
    n_quantiles: int = 10,
    long_q: int = 10,
    short_q: int = 1,
) -> pd.Series:
    """Equal-weight long-short portfolio return per month.

    Within each month, assign stocks to quantile buckets based on *signal*.
    Return = mean(long bucket) − mean(short bucket).
    """
    tmp = df[[date_col, ret_col]].copy()
    if isinstance(signal, str):
        tmp["_sig"] = df[signal]
    else:
        tmp["_sig"] = signal.values

    tmp = tmp.dropna(subset=["_sig", ret_col])

    def _ls(g: pd.DataFrame) -> float:
        if len(g) < n_quantiles:
            return np.nan
        g = g.copy()
        g["_q"] = pd.qcut(g["_sig"], n_quantiles, labels=False, duplicates="drop") + 1
        max_q = g["_q"].max()
        min_q = g["_q"].min()
        long_ret = g.loc[g["_q"] == max_q, ret_col].mean()
        short_ret = g.loc[g["_q"] == min_q, ret_col].mean()
        return long_ret - short_ret

    ls_ret = tmp.groupby(date_col).apply(_ls, include_groups=False)
    ls_ret.name = "ls_ret"
    return ls_ret.dropna()


def zscore_weighted_portfolio_returns(
    df: pd.DataFrame,
    signal: str | pd.Series,
    ret_col: str = "ret",
    date_col: str = "yyyymm",
) -> pd.Series:
    """Z-score weighted long-short portfolio (dollar-neutral)."""
    tmp = df[[date_col, ret_col]].copy()
    if isinstance(signal, str):
        tmp["_sig"] = df[signal]
    else:
        tmp["_sig"] = signal.values
    tmp = tmp.dropna(subset=["_sig", ret_col])

    def _ws(g: pd.DataFrame) -> float:
        z = (g["_sig"] - g["_sig"].mean())
        denom = g["_sig"].std()
        if denom == 0 or np.isnan(denom):
            return np.nan
        z = z / denom
        # dollar-neutral: weights sum to zero
        return (z * g[ret_col]).sum() / np.abs(z).sum()

    ret = tmp.groupby(date_col).apply(_ws, include_groups=False)
    ret.name = "ls_ret"
    return ret.dropna()


# ── metrics ───────────────────────────────────────────────────────────────────

def annualized_sharpe(monthly_ret: pd.Series) -> float:
    """Annualized Sharpe ratio from monthly excess returns (assumes rf ≈ 0)."""
    if len(monthly_ret) < 2:
        return np.nan
    return float(monthly_ret.mean() / monthly_ret.std() * np.sqrt(12))


def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R² (1 − SS_res / SS_tot with mean-zero benchmark)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)  # benchmark = 0
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)
