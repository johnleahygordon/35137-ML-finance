"""Machine learning model fitting and evaluation for Q1b."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean((y[mask] - yhat[mask]) ** 2))


def _safe_predict(model: Any, X: np.ndarray) -> np.ndarray:
    pred = model.predict(X)
    if pred.ndim > 1:
        pred = pred.ravel()
    return pred


# ── model builders ────────────────────────────────────────────────────────────

def build_ols() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ols", LinearRegression()),
    ])


def build_lasso(alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha, max_iter=10_000)),
    ])


def build_ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def build_elasticnet(alpha: float, l1_ratio: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("en", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000)),
    ])


def build_rbf_lasso(alpha: float, n_components: int, gamma: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(n_components=n_components, gamma=gamma, random_state=seed)),
        ("lasso", Lasso(alpha=alpha, max_iter=10_000)),
    ])


def build_rbf_ridge(alpha: float, n_components: int, gamma: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(n_components=n_components, gamma=gamma, random_state=seed)),
        ("ridge", Ridge(alpha=alpha)),
    ])


def build_rbf_elasticnet(alpha: float, l1_ratio: float,
                         n_components: int, gamma: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(n_components=n_components, gamma=gamma, random_state=seed)),
        ("en", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000)),
    ])


def build_pls(n_components: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pls", PLSRegression(n_components=n_components)),
    ])


def build_rbf_pls(n_components: int, rbf_components: int, gamma: float, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(n_components=rbf_components, gamma=gamma, random_state=seed)),
        ("pls", PLSRegression(n_components=n_components)),
    ])


def build_gbr(n_estimators: int, max_depth: int, learning_rate: float, seed: int) -> Pipeline:
    return Pipeline([
        ("gbr", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            subsample=0.8,
        )),
    ])


# ── grid search on validation ────────────────────────────────────────────────

def fit_and_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
) -> dict[str, dict[str, Any]]:
    """Fit all model families, select best hyper-params on validation, return results.

    Returns dict of {model_name: {"model": fitted_model, "params": {...}, "val_mse": float}}
    """
    results: dict[str, dict[str, Any]] = {}

    # (i) OLS
    logger.info("Fitting OLS ...")
    ols = build_ols()
    ols.fit(X_train, y_train)
    results["OLS"] = {
        "model": ols,
        "params": {},
        "val_mse": _mse(y_val, _safe_predict(ols, X_val)),
    }

    # (ii) Lasso
    logger.info("Fitting Lasso ...")
    best_lasso, best_lasso_mse, best_lasso_a = None, np.inf, None
    for a in cfg.ml.lasso_alphas:
        m = build_lasso(a)
        m.fit(X_train, y_train)
        mse = _mse(y_val, _safe_predict(m, X_val))
        if mse < best_lasso_mse:
            best_lasso, best_lasso_mse, best_lasso_a = m, mse, a
    results["Lasso"] = {"model": best_lasso, "params": {"alpha": best_lasso_a}, "val_mse": best_lasso_mse}

    # (ii) Ridge
    logger.info("Fitting Ridge ...")
    best_ridge, best_ridge_mse, best_ridge_a = None, np.inf, None
    for a in cfg.ml.ridge_alphas:
        m = build_ridge(a)
        m.fit(X_train, y_train)
        mse = _mse(y_val, _safe_predict(m, X_val))
        if mse < best_ridge_mse:
            best_ridge, best_ridge_mse, best_ridge_a = m, mse, a
    results["Ridge"] = {"model": best_ridge, "params": {"alpha": best_ridge_a}, "val_mse": best_ridge_mse}

    # (ii) Elastic Net
    logger.info("Fitting ElasticNet ...")
    best_en, best_en_mse, best_en_params = None, np.inf, {}
    for a in cfg.ml.elasticnet_alphas:
        for l1 in cfg.ml.elasticnet_l1_ratios:
            m = build_elasticnet(a, l1)
            m.fit(X_train, y_train)
            mse = _mse(y_val, _safe_predict(m, X_val))
            if mse < best_en_mse:
                best_en, best_en_mse = m, mse
                best_en_params = {"alpha": a, "l1_ratio": l1}
    results["ElasticNet"] = {"model": best_en, "params": best_en_params, "val_mse": best_en_mse}

    # (iii) RBF + Lasso
    logger.info("Fitting RBF+Lasso ...")
    best_rl, best_rl_mse, best_rl_a = None, np.inf, None
    for a in cfg.ml.lasso_alphas:
        m = build_rbf_lasso(a, cfg.ml.rbf_n_components, cfg.ml.rbf_gamma, cfg.seed)
        m.fit(X_train, y_train)
        mse = _mse(y_val, _safe_predict(m, X_val))
        if mse < best_rl_mse:
            best_rl, best_rl_mse, best_rl_a = m, mse, a
    results["RBF+Lasso"] = {"model": best_rl, "params": {"alpha": best_rl_a}, "val_mse": best_rl_mse}

    # (iii) RBF + Ridge
    logger.info("Fitting RBF+Ridge ...")
    best_rr, best_rr_mse, best_rr_a = None, np.inf, None
    for a in cfg.ml.ridge_alphas:
        m = build_rbf_ridge(a, cfg.ml.rbf_n_components, cfg.ml.rbf_gamma, cfg.seed)
        m.fit(X_train, y_train)
        mse = _mse(y_val, _safe_predict(m, X_val))
        if mse < best_rr_mse:
            best_rr, best_rr_mse, best_rr_a = m, mse, a
    results["RBF+Ridge"] = {"model": best_rr, "params": {"alpha": best_rr_a}, "val_mse": best_rr_mse}

    # (iii) RBF + ElasticNet
    logger.info("Fitting RBF+ElasticNet ...")
    best_ren, best_ren_mse, best_ren_params = None, np.inf, {}
    for a in cfg.ml.elasticnet_alphas:
        for l1 in cfg.ml.elasticnet_l1_ratios:
            m = build_rbf_elasticnet(a, l1, cfg.ml.rbf_n_components, cfg.ml.rbf_gamma, cfg.seed)
            m.fit(X_train, y_train)
            mse = _mse(y_val, _safe_predict(m, X_val))
            if mse < best_ren_mse:
                best_ren, best_ren_mse = m, mse
                best_ren_params = {"alpha": a, "l1_ratio": l1}
    results["RBF+ElasticNet"] = {"model": best_ren, "params": best_ren_params, "val_mse": best_ren_mse}

    # (iv) PLS on linear
    logger.info("Fitting PLS (linear) ...")
    best_pls, best_pls_mse, best_pls_n = None, np.inf, None
    for n in cfg.ml.pls_components:
        if n > min(X_train.shape[1], X_train.shape[0]):
            continue
        m = build_pls(n)
        m.fit(X_train, y_train)
        mse = _mse(y_val, _safe_predict(m, X_val))
        if mse < best_pls_mse:
            best_pls, best_pls_mse, best_pls_n = m, mse, n
    results["PLS_linear"] = {"model": best_pls, "params": {"n_components": best_pls_n}, "val_mse": best_pls_mse}

    # (iv) PLS on RBF expansion
    logger.info("Fitting PLS (RBF) ...")
    best_rpls, best_rpls_mse, best_rpls_n = None, np.inf, None
    for n in cfg.ml.pls_components:
        if n > cfg.ml.rbf_n_components:
            continue
        m = build_rbf_pls(n, cfg.ml.rbf_n_components, cfg.ml.rbf_gamma, cfg.seed)
        m.fit(X_train, y_train)
        mse = _mse(y_val, _safe_predict(m, X_val))
        if mse < best_rpls_mse:
            best_rpls, best_rpls_mse, best_rpls_n = m, mse, n
    results["PLS_RBF"] = {"model": best_rpls, "params": {"n_components": best_rpls_n}, "val_mse": best_rpls_mse}

    # (v) GBR
    logger.info("Fitting GBR ...")
    best_gbr, best_gbr_mse, best_gbr_params = None, np.inf, {}
    for n_est in cfg.ml.gbr_n_estimators:
        for md in cfg.ml.gbr_max_depths:
            for lr in cfg.ml.gbr_learning_rates:
                m = build_gbr(n_est, md, lr, cfg.seed)
                m.fit(X_train, y_train)
                mse = _mse(y_val, _safe_predict(m, X_val))
                if mse < best_gbr_mse:
                    best_gbr, best_gbr_mse = m, mse
                    best_gbr_params = {"n_estimators": n_est, "max_depth": md, "learning_rate": lr}
    results["GBR"] = {"model": best_gbr, "params": best_gbr_params, "val_mse": best_gbr_mse}

    for name, r in results.items():
        logger.info("  %-20s  val_mse=%.6f  params=%s", name, r["val_mse"], r["params"])

    return results
