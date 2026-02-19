"""Structured feature engineering — Rungs 2 and 3 of the model ladder.

Rung 2 — Theory features:
    - rate_change_bps       : Fed rate change at this meeting
    - fed_minus_X_pre       : Policy spread (Fed − foreign CB) day before meeting
    - spread_change         : Change in spread vs prior meeting

Rung 3 — Cross-asset features (pre-announcement window, no leakage):
    - ust2y_pre_ret_bps     : 2Y Treasury yield change in pre-window (in bps)
    - ust10y_pre_ret_bps    : 10Y Treasury yield change in pre-window
    - slope_pre_change_bps  : Yield curve slope change (10Y - 2Y) in pre-window
    - spx_pre_ret_bps       : S&P 500 log-return in pre-window (bps)
    - vix_pre_level         : VIX level at end of pre-window
    - vix_pre_change        : VIX change in pre-window
    Meeting-level metadata (per-pair indicators added during merge):
    - is_hike, is_cut, is_hold (bool → int)
    - dissent_count
    - rate_change_bps

All features are computed from data STRICTLY before the announcement window
to prevent leakage.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


# ── helpers ───────────────────────────────────────────────────────────────────


def _pre_window_bars(
    bars: pd.DataFrame,
    meeting_id: str,
    source: str,
    ann_date,
    start_hhmm: str,
    end_hhmm: str,
) -> pd.DataFrame:
    """Return bars for source/meeting within the pre-announcement window."""
    t_start = datetime.strptime(start_hhmm, "%H:%M").time()
    t_end = datetime.strptime(end_hhmm, "%H:%M").time()
    try:
        dt_start = ET.localize(datetime.combine(ann_date, t_start), is_dst=None)
        dt_end = ET.localize(datetime.combine(ann_date, t_end), is_dst=None)
    except pytz.exceptions.AmbiguousTimeError:
        dt_start = ET.localize(datetime.combine(ann_date, t_start), is_dst=False)
        dt_end = ET.localize(datetime.combine(ann_date, t_end), is_dst=False)

    mask = (
        (bars["source"] == source)
        & (bars["meeting_id"] == meeting_id)
        & (bars["timestamp_et"] >= dt_start)
        & (bars["timestamp_et"] < dt_end)
    )
    return bars[mask].sort_values("timestamp_et")


def _log_ret_bps(entry: float | None, exit_: float | None) -> float:
    """Log-return in basis points; NaN on bad inputs."""
    if entry is None or exit_ is None or entry <= 0 or exit_ <= 0:
        return np.nan
    return np.log(exit_ / entry) * 10_000


def _level_change(entry: float | None, exit_: float | None) -> float:
    """Arithmetic difference (exit − entry); NaN on bad inputs."""
    if entry is None or exit_ is None:
        return np.nan
    return exit_ - entry


# ── Rung 2: theory features ───────────────────────────────────────────────────


def make_theory_features(
    fomc_meta: pd.DataFrame,
    policy_rates: pd.DataFrame,
    pair_cb_map: dict[str, str],
) -> pd.DataFrame:
    """Build Rung-2 theory features — one row per (meeting_id, pair).

    pair_cb_map: {"USDEUR": "ECB", "USDJPY": "BOJ", ...}
    """
    # Map CB name → policy rate column
    cb_col = {
        "ECB": "ecb_rate",
        "BOJ": "boj_rate",
        "BOE": "boe_rate",
        "BOC": "boc_rate",
    }
    spread_col = {
        "ECB": "fed_minus_ecb",
        "BOJ": "fed_minus_boj",
        "BOE": "fed_minus_boe",
        "BOC": "fed_minus_boc",
    }

    rows: list[dict] = []

    for _, meta in fomc_meta.iterrows():
        mid = meta["meeting_id"]
        ann_date = meta["announcement_et"].date()
        rate_change_bps = (meta["rate_change"] * 100) if not pd.isna(meta["rate_change"]) else np.nan

        # Policy spread as of the day before announcement (no leakage)
        prev_day = pd.Timestamp(ann_date) - pd.Timedelta(days=1)
        # Align to available dates (policy_rates may not have every day)
        available = policy_rates[policy_rates.index <= prev_day]
        if available.empty:
            spread_row = None
        else:
            spread_row = available.iloc[-1]

        for pair, cb in pair_cb_map.items():
            spread_pre = float(spread_row[spread_col[cb]]) if spread_row is not None and not pd.isna(spread_row[spread_col[cb]]) else np.nan

            # Change in spread vs previous meeting
            prev_meetings = fomc_meta[fomc_meta["announcement_et"] < meta["announcement_et"]]
            if prev_meetings.empty:
                spread_change = np.nan
            else:
                prev_ann_date = prev_meetings.iloc[-1]["announcement_et"].date()
                prev_prev_day = pd.Timestamp(prev_ann_date) - pd.Timedelta(days=1)
                prev_avail = policy_rates[policy_rates.index <= prev_prev_day]
                if prev_avail.empty:
                    spread_change = np.nan
                else:
                    prev_spread_pre = float(prev_avail.iloc[-1][spread_col[cb]])
                    spread_change = spread_pre - prev_spread_pre

            rows.append(
                {
                    "meeting_id": mid,
                    "pair": pair,
                    "rate_change_bps": rate_change_bps,
                    "fed_minus_foreign_pre": spread_pre,
                    "spread_change": spread_change,
                    "is_hike": int(meta["is_hike"]),
                    "is_cut": int(meta["is_cut"]),
                    "is_hold": int(meta["is_hold"]),
                    "dissent_count": int(meta["votes_against"]),
                }
            )

    df = pd.DataFrame(rows)
    logger.info("Theory features: %d rows (meeting × pair)", len(df))
    return df


# ── Rung 3: cross-asset pre-announcement features ─────────────────────────────


def make_cross_asset_features(
    bars: pd.DataFrame,
    fomc_meta: pd.DataFrame,
    pre_start: str = "10:00",
    pre_end: str = "13:55",
) -> pd.DataFrame:
    """Build Rung-3 cross-asset features — one row per meeting_id.

    Uses bars in the pre-announcement window (default 10:00–13:55 ET)
    to ensure no leakage into either prediction window.
    """
    rows: list[dict] = []

    for _, meta in fomc_meta.iterrows():
        mid = meta["meeting_id"]
        ann_date = meta["announcement_et"].date()

        def _get_window(source: str) -> pd.DataFrame:
            return _pre_window_bars(bars, mid, source, ann_date, pre_start, pre_end)

        # 2Y Treasury yield
        ust2y = _get_window("UST2Y")
        if len(ust2y) >= 2:
            ust2y_ret = _level_change(ust2y["close"].iloc[0], ust2y["close"].iloc[-1])
        else:
            ust2y_ret = np.nan

        # 10Y Treasury yield
        ust10y = _get_window("UST10Y")
        if len(ust10y) >= 2:
            ust10y_ret = _level_change(ust10y["close"].iloc[0], ust10y["close"].iloc[-1])
            slope_start = (ust10y["close"].iloc[0] - ust2y["close"].iloc[0]) if len(ust2y) >= 2 else np.nan
            slope_end = (ust10y["close"].iloc[-1] - ust2y["close"].iloc[-1]) if len(ust2y) >= 2 else np.nan
            slope_change = _level_change(slope_start, slope_end)
        else:
            ust10y_ret = np.nan
            slope_change = np.nan

        # S&P 500 (log-return in bps)
        spx = _get_window("SPX")
        if len(spx) >= 2:
            spx_ret = _log_ret_bps(spx["close"].iloc[0], spx["close"].iloc[-1])
        else:
            spx_ret = np.nan

        # VIX level and change
        vix = _get_window("VIX")
        if len(vix) >= 2:
            vix_level = float(vix["close"].iloc[-1])
            vix_change = _level_change(vix["close"].iloc[0], vix["close"].iloc[-1])
        elif len(vix) == 1:
            vix_level = float(vix["close"].iloc[0])
            vix_change = np.nan
        else:
            vix_level = np.nan
            vix_change = np.nan

        rows.append(
            {
                "meeting_id": mid,
                "ust2y_pre_chg": ust2y_ret,        # in yield points (e.g. 0.01 = 1bp)
                "ust10y_pre_chg": ust10y_ret,
                "slope_pre_change": slope_change,
                "spx_pre_ret_bps": spx_ret,
                "vix_pre_level": vix_level,
                "vix_pre_change": vix_change,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Cross-asset features: %d rows (one per meeting)", len(df))
    return df


# ── Combine ───────────────────────────────────────────────────────────────────


def build_structured_features(
    fomc_meta: pd.DataFrame,
    policy_rates: pd.DataFrame,
    bars: pd.DataFrame,
    pair_cb_map: dict[str, str],
    pre_start: str = "10:00",
    pre_end: str = "13:55",
) -> pd.DataFrame:
    """Merge theory + cross-asset features into a single (meeting_id, pair) DataFrame."""
    theory = make_theory_features(fomc_meta, policy_rates, pair_cb_map)
    cross_asset = make_cross_asset_features(bars, fomc_meta, pre_start, pre_end)

    # cross_asset is meeting-level; merge onto theory (which is meeting × pair)
    df = theory.merge(cross_asset, on="meeting_id", how="left")

    logger.info("Structured features: %d rows, %d columns", len(df), df.shape[1])
    return df
