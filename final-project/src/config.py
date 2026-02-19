"""Centralised configuration loader."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class PathsCfg:
    data_raw: str = "data-raw"
    data_clean: str = "data-clean"
    outputs: str = "outputs"


@dataclass
class WindowCfg:
    start: str = "14:00"
    end: str = "14:30"
    label: str = ""


@dataclass
class WindowsCfg:
    statement: WindowCfg = field(
        default_factory=lambda: WindowCfg("14:00", "14:30", "Statement Reaction")
    )
    digestion: WindowCfg = field(
        default_factory=lambda: WindowCfg("14:30", "16:00", "Post-Statement Digestion")
    )


@dataclass
class PreWindowCfg:
    start: str = "10:00"
    end: str = "13:55"


@dataclass
class CVCfg:
    method: str = "LOO"
    n_splits: int = 41


@dataclass
class ModelsCfg:
    lasso_alphas: list[float] = field(default_factory=list)
    ridge_alphas: list[float] = field(default_factory=list)
    elasticnet_alphas: list[float] = field(default_factory=list)
    elasticnet_l1_ratios: list[float] = field(default_factory=list)
    gbr_n_estimators: list[int] = field(default_factory=list)
    gbr_max_depths: list[int] = field(default_factory=list)
    gbr_learning_rates: list[float] = field(default_factory=list)


@dataclass
class TextCfg:
    llm_model: str = "claude-opus-4-6"
    prompt_version: str = "v1"
    embedding_dim_min: int = 5
    embedding_dim_max: int = 20


@dataclass
class StrategyCfg:
    transaction_cost_bps: int = 2


@dataclass
class Config:
    seed: int = 42
    timezone: str = "America/New_York"
    pairs: list[str] = field(default_factory=list)
    pair_cb_map: dict[str, str] = field(default_factory=dict)
    paths: PathsCfg = field(default_factory=PathsCfg)
    windows: WindowsCfg = field(default_factory=WindowsCfg)
    pre_window: PreWindowCfg = field(default_factory=PreWindowCfg)
    cv: CVCfg = field(default_factory=CVCfg)
    models: ModelsCfg = field(default_factory=ModelsCfg)
    text: TextCfg = field(default_factory=TextCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)


def load_config(path: str | pathlib.Path = "configs/config.yaml") -> Config:
    """Load YAML config and return a Config dataclass."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    w = raw.get("windows", {})
    pw = raw.get("pre_window", {})
    m = raw.get("models", {})
    t = raw.get("text", {})

    return Config(
        seed=raw.get("seed", 42),
        timezone=raw.get("timezone", "America/New_York"),
        pairs=raw.get("pairs", []),
        pair_cb_map=raw.get("pair_cb_map", {}),
        paths=PathsCfg(
            data_raw=raw.get("paths", {}).get("data_raw", "data-raw"),
            data_clean=raw.get("paths", {}).get("data_clean", "data-clean"),
            outputs=raw.get("paths", {}).get("outputs", "outputs"),
        ),
        windows=WindowsCfg(
            statement=WindowCfg(
                start=w.get("statement", {}).get("start", "14:00"),
                end=w.get("statement", {}).get("end", "14:30"),
                label=w.get("statement", {}).get("label", "Statement Reaction"),
            ),
            digestion=WindowCfg(
                start=w.get("digestion", {}).get("start", "14:30"),
                end=w.get("digestion", {}).get("end", "16:00"),
                label=w.get("digestion", {}).get("label", "Post-Statement Digestion"),
            ),
        ),
        pre_window=PreWindowCfg(
            start=pw.get("start", "10:00"),
            end=pw.get("end", "13:55"),
        ),
        cv=CVCfg(
            method=raw.get("cv", {}).get("method", "LOO"),
            n_splits=raw.get("cv", {}).get("n_splits", 41),
        ),
        models=ModelsCfg(
            lasso_alphas=m.get("lasso_alphas", []),
            ridge_alphas=m.get("ridge_alphas", []),
            elasticnet_alphas=m.get("elasticnet_alphas", []),
            elasticnet_l1_ratios=m.get("elasticnet_l1_ratios", []),
            gbr_n_estimators=m.get("gbr_n_estimators", []),
            gbr_max_depths=m.get("gbr_max_depths", []),
            gbr_learning_rates=m.get("gbr_learning_rates", []),
        ),
        text=TextCfg(
            llm_model=t.get("llm_model", "claude-opus-4-6"),
            prompt_version=t.get("prompt_version", "v1"),
            embedding_dim_min=t.get("embedding_dim_min", 5),
            embedding_dim_max=t.get("embedding_dim_max", 20),
        ),
        strategy=StrategyCfg(
            transaction_cost_bps=raw.get("strategy", {}).get("transaction_cost_bps", 2),
        ),
    )
