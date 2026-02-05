"""Data loading, schema inference, and time-split utilities."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_parquet(path: str) -> pd.DataFrame:
    """Load a parquet file, coerce the return column to float, and report."""
    df = pd.read_parquet(path)
    logger.info("Loaded %s  shape=%s", path, df.shape)
    return df


def load_lsret(path: str) -> pd.DataFrame:
    """Load lsret CSV, parse dates, sort."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %s  shape=%s  date range %s – %s",
                path, df.shape, df["date"].min(), df["date"].max())
    return df


# ── schema inference ──────────────────────────────────────────────────────────

def infer_schema(df: pd.DataFrame,
                 id_col: str = "permno",
                 date_col: str = "yyyymm",
                 ret_col: str = "ret") -> dict[str, Any]:
    """Detect id / date / return / feature columns programmatically."""
    schema: dict[str, Any] = {"id": id_col, "date": date_col, "ret": ret_col}

    features = [c for c in df.columns if c not in {id_col, date_col, ret_col}]
    schema["features"] = features
    schema["n_features"] = len(features)
    schema["n_rows"] = len(df)
    schema["n_ids"] = df[id_col].nunique() if id_col in df.columns else None
    schema["date_range"] = (df[date_col].min(), df[date_col].max()) if date_col in df.columns else None

    logger.info("Schema: %d features, %d rows, %s ids, dates %s",
                schema["n_features"], schema["n_rows"],
                schema["n_ids"], schema["date_range"])
    return schema


# ── data cleaning ─────────────────────────────────────────────────────────────

def clean_panel(df: pd.DataFrame,
                id_col: str,
                date_col: str,
                ret_col: str,
                features: list[str]) -> pd.DataFrame:
    """Clean a stock-month panel: coerce ret, drop non-numeric ret, impute features."""
    df = df.copy()
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[ret_col])
    logger.info("Dropped %d rows with non-numeric/missing return", n_before - len(df))

    # Cross-sectional median imputation per month (vectorised per feature)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for feat in features:
            if df[feat].isna().any():
                medians = df.groupby(date_col)[feat].transform("median")
                df[feat] = df[feat].fillna(medians)
    # Drop any remaining rows that still have all-NaN features
    df = df.dropna(subset=features, how="all")
    logger.info("After cleaning: %d rows", len(df))
    return df


# ── time splits ───────────────────────────────────────────────────────────────

def _yyyymm_to_year(s: pd.Series) -> pd.Series:
    """Convert yyyymm integer to year integer."""
    return s // 100


def make_time_splits(
    dates: pd.Series,
    train_years: int = 20,
    val_years: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train / validation / test based on years of data.

    The *first* ``train_years`` calendar years of data form the training set,
    the *next* ``val_years`` form validation, and the remainder is test.
    """
    years = _yyyymm_to_year(dates)
    unique_years = np.sort(years.unique())

    train_end_year = unique_years[min(train_years - 1, len(unique_years) - 1)]
    val_start_year = train_end_year + 1

    val_end_idx = min(train_years + val_years - 1, len(unique_years) - 1)
    val_end_year = unique_years[val_end_idx]

    train_mask = (years <= train_end_year).values
    val_mask = ((years >= val_start_year) & (years <= val_end_year)).values
    test_mask = (years > val_end_year).values

    logger.info(
        "Split  train ≤%d (%d)  val %d–%d (%d)  test >%d (%d)",
        train_end_year, train_mask.sum(),
        val_start_year, val_end_year, val_mask.sum(),
        val_end_year, test_mask.sum(),
    )
    return train_mask, val_mask, test_mask
