"""Question 2: PCA latent factors, indicator replication, max Sharpe."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data import load_lsret
from .plotting import explained_variance_plot, sharpe_vs_k_plot
from .portfolio import annualized_sharpe

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_scale(X_train: np.ndarray, X_test: np.ndarray):
    """StandardScaler that drops zero-variance columns to avoid NaN."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)

    # Identify columns that are constant (zero variance) → scaler produces NaN
    good_cols = np.isfinite(X_tr_s).all(axis=0)
    if not good_cols.all():
        n_bad = (~good_cols).sum()
        logger.debug("Dropping %d zero-variance columns from scaler output", n_bad)
        X_tr_s = X_tr_s[:, good_cols]
        X_te_s = scaler.transform(X_test)[:, good_cols]
    else:
        X_te_s = scaler.transform(X_test)

    return X_tr_s, X_te_s, good_cols


def _build_portfolio_returns_from_chars(
    df: pd.DataFrame,
    schema: dict,
    cfg: Config,
) -> pd.DataFrame:
    """Build wide-format monthly long-short return table from a panel (Q1a style).

    Returns DataFrame: rows=months, columns=characteristics.
    """
    from .portfolio import long_short_portfolio_returns, rank_signal_by_month

    date_col = schema["date"]
    ret_col = schema["ret"]
    features = schema["features"]

    records = {}
    for feat in features:
        if df[feat].isna().mean() > 0.80:
            continue
        sig = rank_signal_by_month(df, feat, date_col=date_col)
        ls = long_short_portfolio_returns(
            df, sig, ret_col=ret_col, date_col=date_col,
            n_quantiles=cfg.portfolio.n_quantiles,
        )
        if len(ls) < 24:
            continue
        records[feat] = ls

    wide = pd.DataFrame(records)
    wide.index.name = date_col
    return wide


def _get_years(wide_clean: pd.DataFrame):
    """Extract year from index (handles both DatetimeIndex and int yyyymm)."""
    idx = wide_clean.index
    if hasattr(idx, 'year') and not isinstance(idx, pd.RangeIndex):
        return pd.Series(idx.year, index=idx)
    else:
        return pd.Series(idx // 100, index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# Q2(a): PCA latent factors
# ═══════════════════════════════════════════════════════════════════════════════

def run_q2a(
    lsret: pd.DataFrame,
    large_wide: pd.DataFrame,
    small_wide: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> dict:
    """Run PCA on three portfolio sets, report explained variance, loadings, Sharpes."""
    results = {}

    datasets = {
        "lsret": lsret,
        "large_char": large_wide,
        "small_char": small_wide,
    }

    all_factor_sharpes = {}

    for name, wide_df in datasets.items():
        logger.info("Q2a: PCA on %s (%d periods x %d portfolios)", name, *wide_df.shape)

        # Drop columns/rows with all NaN, then fill remaining with 0
        wide_clean = wide_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        wide_clean = wide_clean.fillna(0)

        # Drop zero-variance columns
        var = wide_clean.var()
        wide_clean = wide_clean.loc[:, var > 0]

        if wide_clean.shape[1] < 2:
            logger.warning("Q2a: %s has too few columns for PCA, skipping", name)
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(wide_clean.values)
        # Replace any remaining NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(X)
        explained = pca.explained_variance_ratio_

        # Save explained variance plot
        explained_variance_plot(
            explained,
            title=f"Q2a: Cumulative Explained Variance — {name}",
            save_path=out_dir / f"q2a_explained_variance_{name}.png",
        )

        # Threshold counts
        cumvar = np.cumsum(explained)
        thresholds = {}
        for t in [0.80, 0.90, 0.95]:
            thresholds[f"{t:.0%}"] = int(np.searchsorted(cumvar, t) + 1)
        logger.info("Q2a %s: factors for 80%%=%d, 90%%=%d, 95%%=%d",
                     name, thresholds["80%"], thresholds["90%"], thresholds["95%"])

        # Top loadings per factor (first 5 factors)
        n_show = min(5, pca.components_.shape[0])
        loadings_records = []
        for i in range(n_show):
            comp = pca.components_[i]
            top_idx = np.argsort(np.abs(comp))[::-1][:10]
            for j in top_idx:
                loadings_records.append({
                    "dataset": name,
                    "factor": i + 1,
                    "portfolio": wide_clean.columns[j],
                    "loading": comp[j],
                })
        loadings_df = pd.DataFrame(loadings_records)
        loadings_df.to_csv(out_dir / f"q2a_factor_loadings_{name}.csv", index=False)

        # Factor Sharpes
        factor_sharpes = {}
        for i in range(n_show):
            factor_ret = pd.Series(factors[:, i], index=wide_clean.index)
            factor_sharpes[f"Factor_{i+1}"] = annualized_sharpe(factor_ret)
        all_factor_sharpes[name] = factor_sharpes

        # Portfolio-level Sharpes
        port_sharpes = {col: annualized_sharpe(wide_clean[col].dropna()) for col in wide_clean.columns}

        results[name] = {
            "pca": pca,
            "scaler": scaler,
            "factors": factors,
            "explained": explained,
            "thresholds": thresholds,
            "factor_sharpes": factor_sharpes,
            "port_sharpes": port_sharpes,
            "wide_clean": wide_clean,
        }

    # Save combined factor sharpes
    factor_sharpe_df = pd.DataFrame(all_factor_sharpes)
    factor_sharpe_df.to_csv(out_dir / "q2a_factor_sharpes.csv")
    logger.info("Q2a factor Sharpes:\n%s", factor_sharpe_df)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Q2(b): "Predict the indicator" via ridge/lasso pre-2004
# ═══════════════════════════════════════════════════════════════════════════════

def run_q2b(
    wide_df: pd.DataFrame,
    cutoff_year: int,
    ridge_alphas: list[float],
    lasso_alphas: list[float],
    label: str,
    out_dir: Path,
) -> dict[str, float]:
    """Predict indicator=1, use coefficients as portfolio weights OOS.

    Uses raw returns (no standardisation) with fit_intercept=False.
    This is the Britten-Jones (1999) approach: regressing 1 on returns
    without intercept yields regularised mean-variance weights.

    wide_df: rows=date, columns=portfolio returns.
    Returns dict of {method: annualized Sharpe OOS}.
    """
    wide_clean = wide_df.dropna(axis=1, how="all").fillna(0)
    var = wide_clean.var()
    wide_clean = wide_clean.loc[:, var > 0]

    years = _get_years(wide_clean)
    train_mask = years < cutoff_year
    test_mask = years >= cutoff_year

    X_train = wide_clean.loc[train_mask].values
    X_test = wide_clean.loc[test_mask].values

    if X_train.shape[0] < 10 or X_test.shape[0] < 10:
        logger.warning("Q2b (%s): insufficient data (train=%d, test=%d)",
                       label, X_train.shape[0], X_test.shape[0])
        return {}

    logger.info("Q2b (%s): train=%d, test=%d, features=%d",
                label, X_train.shape[0], X_test.shape[0], X_train.shape[1])

    y_train = np.ones(X_train.shape[0])

    results: dict[str, float] = {}

    # Ridge: select alpha on validation (last 20% of training)
    n_val = max(1, int(X_train.shape[0] * 0.2))
    X_tr, X_va = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_va = y_train[:-n_val], y_train[-n_val:]

    best_ridge_a, best_ridge_mse = None, np.inf
    for a in ridge_alphas:
        m = Ridge(alpha=a, fit_intercept=False)
        m.fit(X_tr, y_tr)
        mse = float(np.mean((y_va - m.predict(X_va)) ** 2))
        if mse < best_ridge_mse:
            best_ridge_a, best_ridge_mse = a, mse

    ridge = Ridge(alpha=best_ridge_a, fit_intercept=False)
    ridge.fit(X_train, y_train)
    weights_ridge = ridge.coef_

    abs_sum = np.sum(np.abs(weights_ridge))
    if abs_sum > 0:
        oos_ret_ridge = X_test @ weights_ridge / abs_sum
        sr_ridge = annualized_sharpe(pd.Series(oos_ret_ridge))
    else:
        sr_ridge = np.nan
    results[f"Ridge (alpha={best_ridge_a})"] = sr_ridge

    # Lasso
    best_lasso_a, best_lasso_mse = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in lasso_alphas:
            m = Lasso(alpha=a, max_iter=50_000, fit_intercept=False)
            m.fit(X_tr, y_tr)
            mse = float(np.mean((y_va - m.predict(X_va)) ** 2))
            if mse < best_lasso_mse:
                best_lasso_a, best_lasso_mse = a, mse

        lasso = Lasso(alpha=best_lasso_a, max_iter=50_000, fit_intercept=False)
        lasso.fit(X_train, y_train)
    weights_lasso = lasso.coef_

    abs_sum_l = np.sum(np.abs(weights_lasso))
    if abs_sum_l > 0:
        oos_ret_lasso = X_test @ weights_lasso / abs_sum_l
        sr_lasso = annualized_sharpe(pd.Series(oos_ret_lasso))
    else:
        sr_lasso = np.nan
    results[f"Lasso (alpha={best_lasso_a})"] = sr_lasso

    logger.info("Q2b (%s): Ridge Sharpe=%.3f (alpha=%.4f), Lasso Sharpe=%.3f (alpha=%.6f)",
                label, sr_ridge, best_ridge_a, sr_lasso, best_lasso_a)

    result_df = pd.DataFrame([
        {"method": k, "sharpe": v, "dataset": label} for k, v in results.items()
    ])
    result_df.to_csv(out_dir / f"q2b_indicator_{label}.csv", index=False)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Q2(c): PCA compression + indicator for range of k
# ═══════════════════════════════════════════════════════════════════════════════

def run_q2c(
    wide_df: pd.DataFrame,
    cutoff_year: int,
    ridge_alphas: list[float],
    lasso_alphas: list[float],
    k_max: int,
    label: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Repeat Q2b after PCA compression for k=1..k_max.

    PCA is fit on training-period raw returns, then the indicator regression
    is performed on the resulting factors with fit_intercept=False.
    """
    wide_clean = wide_df.dropna(axis=1, how="all").fillna(0)
    var = wide_clean.var()
    wide_clean = wide_clean.loc[:, var > 0]

    years = _get_years(wide_clean)
    train_mask = years < cutoff_year
    test_mask = years >= cutoff_year

    X_train_raw = wide_clean.loc[train_mask].values
    X_test_raw = wide_clean.loc[test_mask].values

    if X_train_raw.shape[0] < 10 or X_test_raw.shape[0] < 10:
        logger.warning("Q2c (%s): insufficient data", label)
        return pd.DataFrame()

    y_train = np.ones(X_train_raw.shape[0])
    n_val = max(1, int(X_train_raw.shape[0] * 0.2))

    actual_k_max = min(k_max, X_train_raw.shape[1], X_train_raw.shape[0] - 1)
    k_values = list(range(1, actual_k_max + 1))

    logger.info("Q2c (%s): sweeping k=1..%d", label, actual_k_max)

    records = []
    for k in k_values:
        pca = PCA(n_components=k)
        F_train = pca.fit_transform(X_train_raw)
        F_test = pca.transform(X_test_raw)

        F_tr, F_va = F_train[:-n_val], F_train[-n_val:]
        y_tr, y_va = y_train[:-n_val], y_train[-n_val:]

        for method_name, alphas, ModelClass in [
            ("Ridge", ridge_alphas, Ridge),
            ("Lasso", lasso_alphas, Lasso),
        ]:
            best_a, best_mse = None, np.inf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for a in alphas:
                    kw = {"alpha": a, "fit_intercept": False}
                    if ModelClass == Lasso:
                        kw["max_iter"] = 50_000
                    m = ModelClass(**kw)
                    m.fit(F_tr, y_tr)
                    mse = float(np.mean((y_va - m.predict(F_va)) ** 2))
                    if mse < best_mse:
                        best_a, best_mse = a, mse

                kw = {"alpha": best_a, "fit_intercept": False}
                if ModelClass == Lasso:
                    kw["max_iter"] = 50_000
                m = ModelClass(**kw)
                m.fit(F_train, y_train)
            w = m.coef_
            abs_sum = np.sum(np.abs(w))
            if abs_sum > 0:
                oos_ret = F_test @ w / abs_sum
                sr = annualized_sharpe(pd.Series(oos_ret))
            else:
                sr = np.nan

            records.append({"k": k, "method": method_name, "sharpe": sr, "alpha": best_a})

    results_df = pd.DataFrame(records)
    results_df["dataset"] = label
    results_df.to_csv(out_dir / f"q2c_sharpe_vs_k_{label}.csv", index=False)

    # Plot
    sharpe_dict = {}
    for method in results_df["method"].unique():
        sub = results_df[results_df["method"] == method]
        sharpe_dict[method] = sub["sharpe"].tolist()

    sharpe_vs_k_plot(
        k_values[:len(sharpe_dict.get("Ridge", []))],
        sharpe_dict,
        title=f"Q2c: OOS Sharpe vs # PCA Factors — {label}",
        save_path=out_dir / f"q2c_sharpe_vs_k_{label}.png",
    )

    logger.info("Q2c (%s): done. Best Ridge Sharpe=%.3f, Best Lasso Sharpe=%.3f",
                label,
                results_df.loc[results_df["method"] == "Ridge", "sharpe"].max(),
                results_df.loc[results_df["method"] == "Lasso", "sharpe"].max())

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# Q2(e): Max Sharpe attempt using lsret
# ═══════════════════════════════════════════════════════════════════════════════

def run_q2e(
    wide_df: pd.DataFrame,
    cutoff_year: int,
    ridge_alphas: list[float],
    k_max: int,
    out_dir: Path,
) -> float:
    """Best-effort max Sharpe using lsret portfolios.

    Strategy: PCA on raw returns + ridge indicator (no intercept),
    with optimal k chosen via validation Sharpe, then volatility scaling.
    """
    wide_clean = wide_df.dropna(axis=1, how="all").fillna(0)
    var = wide_clean.var()
    wide_clean = wide_clean.loc[:, var > 0]

    years = _get_years(wide_clean)
    train_mask = years < cutoff_year
    test_mask = years >= cutoff_year

    X_train_raw = wide_clean.loc[train_mask].values
    X_test_raw = wide_clean.loc[test_mask].values
    test_index = wide_clean.loc[test_mask].index

    if X_train_raw.shape[0] < 10 or X_test_raw.shape[0] < 10:
        logger.warning("Q2e: insufficient data")
        return np.nan

    y_train = np.ones(X_train_raw.shape[0])
    n_val = max(1, int(X_train_raw.shape[0] * 0.2))

    actual_k_max = min(k_max, X_train_raw.shape[1], X_train_raw.shape[0] - 1)

    # Search over k and alpha simultaneously
    best_val_sr, best_k, best_a = -np.inf, None, None
    best_oos_ret = None

    for k in range(1, actual_k_max + 1):
        pca = PCA(n_components=k)
        F_train = pca.fit_transform(X_train_raw)
        F_test = pca.transform(X_test_raw)

        F_tr, F_va = F_train[:-n_val], F_train[-n_val:]
        y_tr, y_va = y_train[:-n_val], y_train[-n_val:]

        for a in ridge_alphas:
            m = Ridge(alpha=a, fit_intercept=False)
            m.fit(F_tr, y_tr)
            w_va = m.coef_
            abs_sum = np.sum(np.abs(w_va))
            if abs_sum == 0:
                continue
            val_ret = F_va @ w_va / abs_sum
            val_std = np.std(val_ret)
            if val_std == 0:
                continue
            val_sr = float(np.mean(val_ret) / val_std * np.sqrt(12))

            if val_sr > best_val_sr:
                # Refit on full training
                m_full = Ridge(alpha=a, fit_intercept=False)
                m_full.fit(F_train, y_train)
                w = m_full.coef_
                abs_sum_full = np.sum(np.abs(w))
                if abs_sum_full > 0:
                    oos_ret = F_test @ w / abs_sum_full
                    best_val_sr = val_sr
                    best_k = k
                    best_a = a
                    best_oos_ret = oos_ret

    if best_oos_ret is None:
        logger.warning("Q2e: no valid strategy found")
        return np.nan

    # Volatility scaling
    oos_series = pd.Series(best_oos_ret, index=test_index)
    target_vol = 0.15
    rolling_vol = oos_series.rolling(12, min_periods=6).std() * np.sqrt(12)
    scale = target_vol / rolling_vol.shift(1)
    scale = scale.clip(0.5, 3.0)
    scaled = oos_series * scale
    scaled = scaled.dropna()

    raw_sr = annualized_sharpe(pd.Series(best_oos_ret))
    final_sr = annualized_sharpe(scaled)

    result = pd.DataFrame({
        "strategy": ["PCA+Ridge+VolScale"],
        "k": [best_k],
        "alpha": [best_a],
        "sharpe_raw": [raw_sr],
        "sharpe_volscaled": [final_sr],
    })
    result.to_csv(out_dir / "q2e_max_sharpe.csv", index=False)

    logger.info("Q2e: best k=%d, alpha=%.4f, raw Sharpe=%.3f, vol-scaled Sharpe=%.3f",
                best_k, best_a, raw_sr, final_sr)

    return final_sr
