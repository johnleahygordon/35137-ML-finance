"""Model ladder: all four rungs of increasing complexity.

Rung 1 — Naive baselines (predict-zero, historical mean)
Rung 2 — OLS on theory features
Rung 3a — Regularised linear (Ridge, LASSO, ElasticNet, Huber) on all structured features
Rung 3b — Gradient-boosted trees on all structured features
Rung 4  — Best rung-3 model augmented with text features (keyword + LLM + embeddings)

All models are wrapped in sklearn Pipelines with StandardScaler.
Hyperparameter selection uses leave-one-meeting-out cross-validation.
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Rung 1: Naive baselines ───────────────────────────────────────────────────


class PredictZero(BaseEstimator, RegressorMixin):
    """Always predicts 0 (no-change benchmark)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class HistoricalMean(BaseEstimator, RegressorMixin):
    """Predicts the expanding mean of past observations.

    Intended for use in LOO-CV where it is fit on the training set and
    then predicts a single constant (the training mean) for the test meeting.
    """

    def fit(self, X, y):
        self.mean_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# ── Model builders (sklearn Pipelines) ───────────────────────────────────────


def _pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_predict_zero() -> BaseEstimator:
    return PredictZero()


def build_historical_mean() -> BaseEstimator:
    return HistoricalMean()


def build_ols() -> Pipeline:
    return _pipeline(LinearRegression())


def build_ridge(alpha: float = 1.0) -> Pipeline:
    return _pipeline(Ridge(alpha=alpha))


def build_lasso(alpha: float = 0.01) -> Pipeline:
    return _pipeline(Lasso(alpha=alpha, max_iter=5000))


def build_elasticnet(alpha: float = 0.01, l1_ratio: float = 0.5) -> Pipeline:
    return _pipeline(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000))


def build_huber(epsilon: float = 1.35, alpha: float = 0.01) -> Pipeline:
    return _pipeline(HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=500))


def build_gbr(
    n_estimators: int = 100,
    max_depth: int = 2,
    learning_rate: float = 0.05,
) -> Pipeline:
    return _pipeline(
        GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )
    )


# ── Feature column definitions ────────────────────────────────────────────────

FEATURE_SETS = {
    "rung2_theory": [
        "rate_change_bps",
        "fed_minus_foreign_pre",
        "spread_change",
        "is_hike",
        "is_cut",
        "is_hold",
        "dissent_count",
    ],
    "rung3_structured": [
        "rate_change_bps",
        "fed_minus_foreign_pre",
        "spread_change",
        "is_hike",
        "is_cut",
        "is_hold",
        "dissent_count",
        "ust2y_pre_chg",
        "ust10y_pre_chg",
        "slope_pre_change",
        "spx_pre_ret_bps",
        "vix_pre_level",
        "vix_pre_change",
    ],
    "rung4_text_kw": [
        "net_hawkish",
        "net_hawkish_norm",
        "balance_sheet_kw",
        "uncertainty_kw",
    ],
    "rung4_text_llm": [
        "hawkish_dovish",
        "inflation_focus",
        "labor_focus",
        "recession_risk",
        "uncertainty_score",
        "forward_guidance_strength",
        "balance_sheet_mention",
    ],
}


def get_feature_cols(panel: pd.DataFrame, feature_set_names: list[str]) -> list[str]:
    """Return all available columns from requested feature sets."""
    cols = []
    for name in feature_set_names:
        if name == "rung4_embeddings":
            emb_cols = [c for c in panel.columns if c.startswith("emb_pc_")]
            cols.extend(emb_cols)
        elif name in FEATURE_SETS:
            cols.extend([c for c in FEATURE_SETS[name] if c in panel.columns])
    return list(dict.fromkeys(cols))  # deduplicate while preserving order


# ── Leave-One-Meeting-Out CV ──────────────────────────────────────────────────


def lomo_cv(
    panel: pd.DataFrame,
    model_builder,
    feature_cols: list[str],
    target_col: str = "log_ret",
    meeting_col: str = "meeting_id",
) -> dict[str, Any]:
    """Leave-One-Meeting-Out cross-validation.

    For each meeting, train on all other meetings and predict.
    Splits by meeting_id (NOT row-level) to avoid data leakage.

    Parameters
    ----------
    panel        : merged panel with features and targets
    model_builder: callable returning a fresh unfitted sklearn estimator
    feature_cols : list of feature column names (must be in panel)
    target_col   : column to predict (e.g. 'log_ret')
    meeting_col  : grouping column

    Returns dict with:
        preds        : Series of OOS predictions indexed like panel
        actuals      : Series of actual values
        meeting_ids  : Series of meeting IDs for each row
        residuals    : actuals − preds
    """
    valid_mask = panel[target_col].notna() & panel[feature_cols].notna().all(axis=1)
    df = panel[valid_mask].copy().reset_index(drop=True)

    meetings = df[meeting_col].unique()
    preds = np.full(len(df), np.nan)

    for test_meeting in meetings:
        test_mask = df[meeting_col] == test_meeting
        train_mask = ~test_mask

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        mdl = model_builder()
        try:
            mdl.fit(X_train, y_train)
            preds[test_mask] = mdl.predict(X_test)
        except Exception as e:
            logger.warning("LOO fit failed for meeting %s: %s", test_meeting, e)
            preds[test_mask] = 0.0

    return {
        "preds": pd.Series(preds, index=df.index),
        "actuals": df[target_col].reset_index(drop=True),
        "meeting_ids": df[meeting_col].reset_index(drop=True),
        "pairs": df["pair"].reset_index(drop=True) if "pair" in df.columns else None,
        "windows": df["window"].reset_index(drop=True) if "window" in df.columns else None,
    }


def grid_search_lomo(
    panel: pd.DataFrame,
    model_builders: dict[str, tuple],  # {name: (builder_fn, param_grid)}
    feature_cols: list[str],
    target_col: str = "log_ret",
    metric: str = "rmse",
) -> tuple[str, dict, float]:
    """Select best hyperparameters using LOO-CV RMSE.

    model_builders: {model_name: (builder_fn, [{param: value}, ...])}
    Returns (best_name, best_params, best_score).
    """
    best_name = None
    best_params = {}
    best_score = np.inf if metric == "rmse" else -np.inf

    for name, (builder_fn, param_grid) in model_builders.items():
        for params in param_grid:
            def _make_model(fn=builder_fn, p=params):
                return fn(**p)

            result = lomo_cv(panel, _make_model, feature_cols, target_col)
            actuals = result["actuals"].values
            preds = result["preds"].values

            valid = ~(np.isnan(actuals) | np.isnan(preds))
            if valid.sum() < 5:
                continue

            if metric == "rmse":
                score = float(np.sqrt(np.mean((actuals[valid] - preds[valid]) ** 2)))
                if score < best_score:
                    best_score = score
                    best_name = f"{name}_{params}"
                    best_params = params
            elif metric == "dir_acc":
                score = float(np.mean(np.sign(actuals[valid]) == np.sign(preds[valid])))
                if score > best_score:
                    best_score = score
                    best_name = f"{name}_{params}"
                    best_params = params

        logger.debug("  %s best so far: %s=%.4f params=%s", name, metric, best_score, best_params)

    return best_name, best_params, best_score
