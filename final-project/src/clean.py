"""Data cleaning, QA, and canonical schema enforcement.

After loading raw data with ingest.py, pass the resulting DataFrames here
to apply quality checks, drop bad rows, and write clean parquet files.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ── Intraday bars QA ──────────────────────────────────────────────────────────


def qa_intraday_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply quality checks to the raw combined intraday bars DataFrame.

    Checks (in order):
        1. Drop rows where close is NaN
        2. Drop rows with zero tick_count (no trades — stale / phantom bar)
        3. Detect and drop duplicate (source, meeting_id, timestamp_et) rows
        4. Flag time gaps > 10 minutes within each (source, meeting_id) group

    Returns
    -------
    df_clean : cleaned DataFrame
    report   : dict with QA statistics for logging / display
    """
    report: dict[str, Any] = {"n_raw": len(df)}

    # 1. Drop missing close
    mask_no_close = df["close"].isna()
    n_no_close = mask_no_close.sum()
    df = df[~mask_no_close].copy()

    # 2. Drop zero-tick bars
    mask_zero_tick = df["tick_count"].notna() & (df["tick_count"] == 0)
    n_zero_tick = mask_zero_tick.sum()
    df = df[~mask_zero_tick].copy()

    # 3. Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["source", "meeting_id", "timestamp_et"])
    n_dupes = n_before - len(df)

    # 4. Gap detection (flag only — do not drop)
    def _max_gap_minutes(ts: pd.Series) -> float:
        ts = ts.sort_values()
        if len(ts) < 2:
            return 0.0
        diffs = ts.diff().dropna()
        return diffs.max().total_seconds() / 60

    gap_report = (
        df.groupby(["source", "meeting_id"])["timestamp_et"]
        .apply(_max_gap_minutes)
        .rename("max_gap_min")
        .reset_index()
    )
    large_gaps = gap_report[gap_report["max_gap_min"] > 10]

    report.update(
        {
            "n_dropped_no_close": int(n_no_close),
            "n_dropped_zero_tick": int(n_zero_tick),
            "n_dropped_duplicates": int(n_dupes),
            "n_clean": len(df),
            "n_large_gaps": len(large_gaps),
            "large_gap_details": large_gaps.to_dict("records"),
        }
    )

    logger.info(
        "QA: %d raw → %d clean  (dropped %d no-close, %d zero-tick, %d dupes)",
        report["n_raw"],
        report["n_clean"],
        n_no_close,
        n_zero_tick,
        n_dupes,
    )
    if len(large_gaps):
        logger.warning("Large gaps (>10 min) in %d source×meeting combinations", len(large_gaps))

    return df.reset_index(drop=True), report


def filter_window(
    bars: pd.DataFrame,
    meeting_date: str,
    start_et: str,
    end_et: str,
    source: str,
) -> pd.DataFrame:
    """Filter bars to a specific ET time window on the FOMC announcement date.

    Parameters
    ----------
    bars        : canonical intraday DataFrame (timestamp_et is tz-aware ET)
    meeting_date: YYYYMMDD string
    start_et    : "HH:MM" window start (inclusive)
    end_et      : "HH:MM" window end (exclusive)
    source      : e.g. "USDEUR"
    """
    import pytz
    from datetime import datetime, time as dt_time

    ET = pytz.timezone("America/New_York")
    date = datetime.strptime(meeting_date, "%Y%m%d").date()

    t_start = datetime.strptime(start_et, "%H:%M").time()
    t_end = datetime.strptime(end_et, "%H:%M").time()

    try:
        dt_start = ET.localize(datetime.combine(date, t_start), is_dst=None)
        dt_end = ET.localize(datetime.combine(date, t_end), is_dst=None)
    except Exception:
        dt_start = ET.localize(datetime.combine(date, t_start), is_dst=False)
        dt_end = ET.localize(datetime.combine(date, t_end), is_dst=False)

    mask = (
        (bars["source"] == source)
        & (bars["meeting_id"] == meeting_date)
        & (bars["timestamp_et"] >= dt_start)
        & (bars["timestamp_et"] < dt_end)
    )
    return bars[mask].copy()


# ── Parquet I/O ───────────────────────────────────────────────────────────────


def write_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write DataFrame to parquet, creating parent dirs as needed.

    Note: tz-aware datetime columns are stored with timezone metadata.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows → %s", len(df), path)


def read_parquet(path: pathlib.Path) -> pd.DataFrame:
    """Read parquet file, restoring timezone-aware columns."""
    df = pd.read_parquet(path)
    logger.info("Read %d rows ← %s", len(df), path)
    return df


# ── Coverage report ───────────────────────────────────────────────────────────


def coverage_report(bars: pd.DataFrame, fomc_meta: pd.DataFrame) -> pd.DataFrame:
    """Return a source × meeting coverage table showing bar counts per combination.

    Useful for the EDA notebook to spot missing files at a glance.
    """
    counts = (
        bars.groupby(["source", "meeting_id"])
        .size()
        .rename("n_bars")
        .reset_index()
    )
    # Pivot: rows = meeting_id, cols = source
    pivot = counts.pivot(index="meeting_id", columns="source", values="n_bars").fillna(0).astype(int)
    # Attach announcement date for readability
    id_date = fomc_meta.set_index("meeting_id")["announcement_et"].dt.date
    pivot = pivot.join(id_date, how="left")
    return pivot.sort_index()
