"""Centralised configuration loader."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class SplitCfg:
    train_years: int = 20
    val_years: int = 12


@dataclass
class PortfolioCfg:
    n_quantiles: int = 10
    long_quantile: int = 10
    short_quantile: int = 1


@dataclass
class MLCfg:
    rbf_n_components: int = 100
    rbf_gamma: float = 1.0
    pls_max_components: int = 20
    lasso_alphas: list[float] = field(default_factory=list)
    ridge_alphas: list[float] = field(default_factory=list)
    elasticnet_alphas: list[float] = field(default_factory=list)
    elasticnet_l1_ratios: list[float] = field(default_factory=list)
    pls_components: list[int] = field(default_factory=list)
    gbr_n_estimators: list[int] = field(default_factory=list)
    gbr_max_depths: list[int] = field(default_factory=list)
    gbr_learning_rates: list[float] = field(default_factory=list)


@dataclass
class Q2Cfg:
    indicator_cutoff_year: int = 2004
    pca_k_range_max: int = 50
    ridge_alphas: list[float] = field(default_factory=list)
    lasso_alphas: list[float] = field(default_factory=list)


@dataclass
class Config:
    seed: int = 42
    large_path: str = "data/largeml.pq"
    small_path: str = "data/smallml.pq"
    lsret_path: str = "data/lsret.csv"
    id_col: str = "permno"
    date_col: str = "yyyymm"
    ret_col: str = "ret"
    output_dir: str = "outputs"
    split: SplitCfg = field(default_factory=SplitCfg)
    portfolio: PortfolioCfg = field(default_factory=PortfolioCfg)
    ml: MLCfg = field(default_factory=MLCfg)
    q2: Q2Cfg = field(default_factory=Q2Cfg)


def load_config(path: str | pathlib.Path = "configs/default.yaml") -> Config:
    """Load YAML config and return a Config dataclass."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    cfg = Config(
        seed=raw.get("seed", 42),
        large_path=raw.get("data", {}).get("large_path", "data/largeml.pq"),
        small_path=raw.get("data", {}).get("small_path", "data/smallml.pq"),
        lsret_path=raw.get("data", {}).get("lsret_path", "data/lsret.csv"),
        id_col=raw.get("schema", {}).get("id_col", "permno"),
        date_col=raw.get("schema", {}).get("date_col", "yyyymm"),
        ret_col=raw.get("schema", {}).get("ret_col", "ret"),
        output_dir=raw.get("outputs", {}).get("dir", "outputs"),
        split=SplitCfg(
            train_years=raw.get("splits", {}).get("train_years", 20),
            val_years=raw.get("splits", {}).get("val_years", 12),
        ),
        portfolio=PortfolioCfg(
            n_quantiles=raw.get("portfolio", {}).get("n_quantiles", 10),
            long_quantile=raw.get("portfolio", {}).get("long_quantile", 10),
            short_quantile=raw.get("portfolio", {}).get("short_quantile", 1),
        ),
        ml=MLCfg(
            rbf_n_components=raw.get("ml", {}).get("rbf_n_components", 100),
            rbf_gamma=raw.get("ml", {}).get("rbf_gamma", 1.0),
            pls_max_components=raw.get("ml", {}).get("pls_max_components", 20),
            lasso_alphas=raw.get("ml", {}).get("lasso_alphas", []),
            ridge_alphas=raw.get("ml", {}).get("ridge_alphas", []),
            elasticnet_alphas=raw.get("ml", {}).get("elasticnet_alphas", []),
            elasticnet_l1_ratios=raw.get("ml", {}).get("elasticnet_l1_ratios", []),
            pls_components=raw.get("ml", {}).get("pls_components", []),
            gbr_n_estimators=raw.get("ml", {}).get("gbr_n_estimators", []),
            gbr_max_depths=raw.get("ml", {}).get("gbr_max_depths", []),
            gbr_learning_rates=raw.get("ml", {}).get("gbr_learning_rates", []),
        ),
        q2=Q2Cfg(
            indicator_cutoff_year=raw.get("q2", {}).get("indicator_cutoff_year", 2004),
            pca_k_range_max=raw.get("q2", {}).get("pca_k_range_max", 50),
            ridge_alphas=raw.get("q2", {}).get("ridge_alphas", []),
            lasso_alphas=raw.get("q2", {}).get("lasso_alphas", []),
        ),
    )
    return cfg
