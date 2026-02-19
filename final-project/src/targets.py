"""Target construction: log-returns for each meeting × pair × window.

Unit of observation: (meeting_id, pair, window)
Maximum panel size: 41 meetings × 4 pairs × 2 windows = 328 rows.

Sign convention (from proposal):
    positive log-return = USD appreciation
    (e.g. USDEUR goes UP → you get more EUR per USD → USD stronger)
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# Minimum number of 5-minute bars required inside a window to compute a return.
MIN_BARS = 3


def _window_return(
    bars: pd.DataFrame,
    meeting_id: str,
    pair: str,
    ann_date,
    start_hhmm: str,
    end_hhmm: str,
) -> dict:
    """Compute log-return for one (meeting, pair, window) cell.

    Uses the last close at or before window_start as the entry price and
    the last close at or before window_end as the exit price.

    Returns a dict with:
        log_ret, abs_ret, direction (int: +1/-1/0), n_bars, has_data (bool)
    """
    t_start = datetime.strptime(start_hhmm, "%H:%M").time()
    t_end = datetime.strptime(end_hhmm, "%H:%M").time()

    try:
        dt_start = ET.localize(datetime.combine(ann_date, t_start), is_dst=None)
        dt_end = ET.localize(datetime.combine(ann_date, t_end), is_dst=None)
    except pytz.exceptions.AmbiguousTimeError:
        dt_start = ET.localize(datetime.combine(ann_date, t_start), is_dst=False)
        dt_end = ET.localize(datetime.combine(ann_date, t_end), is_dst=False)

    # Filter to this source/meeting on the announcement date
    mask = (
        (bars["source"] == pair)
        & (bars["meeting_id"] == meeting_id)
        & (bars["timestamp_et"].dt.date == ann_date)
    )
    day_bars = bars[mask].sort_values("timestamp_et")

    # Bars strictly within [start, end)
    window_bars = day_bars[
        (day_bars["timestamp_et"] >= dt_start) & (day_bars["timestamp_et"] < dt_end)
    ]
    n_bars = len(window_bars)

    if n_bars < MIN_BARS:
        return {
            "log_ret": np.nan,
            "abs_ret": np.nan,
            "direction": np.nan,
            "n_bars": n_bars,
            "has_data": False,
            "price_entry": np.nan,
            "price_exit": np.nan,
        }

    # Entry: last close at or before window start (use the bar just before window)
    pre_bars = day_bars[day_bars["timestamp_et"] <= dt_start]
    price_entry = pre_bars["close"].iloc[-1] if not pre_bars.empty else window_bars["close"].iloc[0]

    # Exit: last close in the window
    price_exit = window_bars["close"].iloc[-1]

    if price_entry is None or price_exit is None or price_entry <= 0 or price_exit <= 0:
        return {
            "log_ret": np.nan,
            "abs_ret": np.nan,
            "direction": np.nan,
            "n_bars": n_bars,
            "has_data": False,
            "price_entry": price_entry,
            "price_exit": price_exit,
        }

    log_ret = np.log(price_exit / price_entry)
    direction = int(np.sign(log_ret)) if not np.isnan(log_ret) else np.nan

    return {
        "log_ret": log_ret,
        "abs_ret": abs(log_ret),
        "direction": direction,
        "n_bars": n_bars,
        "has_data": True,
        "price_entry": price_entry,
        "price_exit": price_exit,
    }


def compute_targets(
    bars: pd.DataFrame,
    fomc_meta: pd.DataFrame,
    pairs: list[str],
    windows: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Build the full (meeting × pair × window) panel with log-return targets.

    Parameters
    ----------
    bars      : clean canonical intraday bars DataFrame
    fomc_meta : FOMC metadata DataFrame (from ingest.load_fomc_metadata)
    pairs     : list of FX pair source names, e.g. ["USDEUR", "USDJPY", ...]
    windows   : {window_name: (start_hhmm, end_hhmm)}, e.g.
                {"statement": ("14:00", "14:30"), "digestion": ("14:30", "16:00")}

    Returns
    -------
    DataFrame with columns:
        meeting_id, pair, window,
        log_ret, abs_ret, direction, n_bars, has_data,
        price_entry, price_exit,
        announcement_et  (for sorting / regime labelling)
    """
    rows: list[dict] = []

    for _, meta_row in fomc_meta.iterrows():
        mid = meta_row["meeting_id"]
        ann_et = meta_row["announcement_et"]
        ann_date = ann_et.date()

        for pair in pairs:
            for window_name, (start_hhmm, end_hhmm) in windows.items():
                result = _window_return(bars, mid, pair, ann_date, start_hhmm, end_hhmm)
                rows.append(
                    {
                        "meeting_id": mid,
                        "pair": pair,
                        "window": window_name,
                        "announcement_et": ann_et,
                        **result,
                    }
                )

    panel = pd.DataFrame(rows)

    n_total = len(panel)
    n_valid = panel["has_data"].sum()
    n_missing = n_total - n_valid

    logger.info(
        "Target panel: %d rows (%d valid, %d missing / flagged)",
        n_total, n_valid, n_missing,
    )
    if n_missing:
        missing_cells = panel[~panel["has_data"]][["meeting_id", "pair", "window"]]
        logger.warning("Missing target cells:\n%s", missing_cells.to_string(index=False))

    return panel.sort_values(["announcement_et", "pair", "window"]).reset_index(drop=True)


def windows_from_config(cfg) -> dict[str, tuple[str, str]]:
    """Extract window dict from a Config object for use with compute_targets."""
    return {
        "statement": (cfg.windows.statement.start, cfg.windows.statement.end),
        "digestion": (cfg.windows.digestion.start, cfg.windows.digestion.end),
    }
