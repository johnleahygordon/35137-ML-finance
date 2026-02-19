"""Raw data ingestion: CSV → DataFrame, PDF → text.

All timestamps are stored as timezone-aware Eastern Time (America/New_York).
Call these functions to obtain raw (un-QA'd) DataFrames; pass results to
clean.py for standardisation and quality checks.
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import re
from datetime import datetime
from typing import Any

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

ET = pytz.timezone("America/New_York")


def _safe_float(s: str) -> float | None:
    """Convert a string to float, returning None for missing / N.A. values."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s in ("N.A.", "N/A", "#N/A N/A", "", "N.A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pct_to_float(s: str) -> float | None:
    """Strip trailing '%' and convert to float (e.g. '0.25%' → 0.25)."""
    if s is None:
        return None
    return _safe_float(s.strip().rstrip("%"))


def _date_to_meeting_id(date_str: str) -> str:
    """Convert M/D/YYYY date string to YYYYMMDD meeting_id (e.g. '1/27/2021' → '20210127')."""
    dt = datetime.strptime(date_str.strip(), "%m/%d/%Y")
    return dt.strftime("%Y%m%d")


# ── FOMC metadata ─────────────────────────────────────────────────────────────


def load_fomc_metadata(root: pathlib.Path) -> pd.DataFrame:
    """Load and parse fomc_meetings.csv.

    Returns a DataFrame with columns:
        meeting_id      str        YYYYMMDD key
        announcement_et datetime   tz-aware ET announcement datetime
        lower_rate      float      lower bound of target range (%)
        upper_rate      float      upper bound of target range (%)
        midpoint        float      (lower + upper) / 2
        rate_change     float      change in midpoint vs previous meeting (NaN for first)
        is_hike         bool
        is_cut          bool
        is_hold         bool
        votes_for       int
        votes_against   int        (= dissent_count)
        policy_concern  str
    """
    fpath = root / "bloomberg-exports" / "fomc_meetings.csv"
    logger.info("Loading FOMC metadata from %s", fpath)

    rows: list[dict] = []
    with open(fpath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meeting_id = _date_to_meeting_id(row["announcement_date"])
            date_str = row["announcement_date"].strip()
            time_str = row["announcement_time"].strip()
            naive_dt = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%Y %H:%M:%S")
            try:
                ann_et = ET.localize(naive_dt, is_dst=None)
            except pytz.exceptions.AmbiguousTimeError:
                ann_et = ET.localize(naive_dt, is_dst=False)

            lower = _pct_to_float(row.get("lower_rate", ""))
            upper = _pct_to_float(row.get("upper_rate", ""))
            midpoint = (lower + upper) / 2 if (lower is not None and upper is not None) else None

            rows.append(
                {
                    "meeting_id": meeting_id,
                    "announcement_et": ann_et,
                    "lower_rate": lower,
                    "upper_rate": upper,
                    "midpoint": midpoint,
                    "votes_for": int(row.get("votes_for", 0) or 0),
                    "votes_against": int(row.get("votes_against", 0) or 0),
                    "policy_concern": row.get("policy_concern", "").strip(),
                }
            )

    df = pd.DataFrame(rows).sort_values("announcement_et").reset_index(drop=True)

    # Rate change vs prior meeting
    df["rate_change"] = df["midpoint"].diff()
    df["is_hike"] = df["rate_change"] > 0
    df["is_cut"] = df["rate_change"] < 0
    df["is_hold"] = df["rate_change"] == 0

    logger.info("Loaded %d FOMC meetings (%s to %s)", len(df), df["meeting_id"].iloc[0], df["meeting_id"].iloc[-1])
    return df


# ── Policy rates ──────────────────────────────────────────────────────────────

# Map Bloomberg ticker prefix → column name
_POLICY_RATE_FILES = {
    "Fed_policy_rate_FEDL01_2021-26.csv": "fed_rate",
    "BOC_policy_rate_OVERNIGHT_2021-26.csv": "boc_rate",
    "BOE_policy_rate_UKBRBASE_2021-26.csv": "boe_rate",
    "BOJ_policy_rate_BOJEPR_2021-26.csv": "boj_rate",
    "ECB_policy_rate_EURR002W_2021-26.csv": "ecb_rate",
}


def load_policy_rates(root: pathlib.Path) -> pd.DataFrame:
    """Load all central bank policy rate CSVs and merge into a single daily DataFrame.

    Returns a DataFrame indexed by date with columns:
        fed_rate, boc_rate, boe_rate, boj_rate, ecb_rate  (all in % terms)
        fed_minus_ecb, fed_minus_boj, fed_minus_boe, fed_minus_boc (spreads)
    """
    rate_dir = root / "bloomberg-exports" / "policy-rate-data"
    logger.info("Loading policy rates from %s", rate_dir)

    series: dict[str, pd.Series] = {}
    for fname, col in _POLICY_RATE_FILES.items():
        fpath = rate_dir / fname
        # The CSV has duplicate column names; read by position (cols 0=Date, 1=PX_LAST)
        tmp = pd.read_csv(fpath, encoding="utf-8-sig", usecols=[0, 1], header=0)
        tmp.columns = ["date", col]
        tmp["date"] = pd.to_datetime(tmp["date"], format="%m/%d/%Y", errors="coerce")
        tmp = tmp.dropna(subset=["date"])
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.set_index("date")[col].sort_index()
        series[col] = tmp
        logger.debug("  %s: %d observations", col, len(tmp))

    df = pd.DataFrame(series)
    # Forward-fill gaps (weekends, holidays)
    df = df.sort_index().ffill()

    # Policy spreads: Fed minus foreign CB (positive = Fed tighter)
    df["fed_minus_ecb"] = df["fed_rate"] - df["ecb_rate"]
    df["fed_minus_boj"] = df["fed_rate"] - df["boj_rate"]
    df["fed_minus_boe"] = df["fed_rate"] - df["boe_rate"]
    df["fed_minus_boc"] = df["fed_rate"] - df["boc_rate"]

    logger.info("Policy rates: %d days, %s to %s", len(df), df.index.min().date(), df.index.max().date())
    return df


# ── Intraday bars ─────────────────────────────────────────────────────────────


def _parse_intraday_csv(filepath: pathlib.Path, meeting_id: str, source: str) -> pd.DataFrame:
    """Parse a single Bloomberg 5-minute intraday bar CSV.

    Bloomberg format:
        Row 1: column header (Time Interval, Close, Net Chg, Open, High, Low, Tick Count, Volume)
        Row 2: Summary row (skip)
        Date rows: "26JAN2021_00:00:00.000000,,,,,,,"  (sets current date context)
        Bar rows:  "17:00 - 17:05,0.8223,-0.0001,..."

    Times in the CSV are Eastern Time. Each file may span 2 calendar days
    (overnight FX session starts the day before the announcement).
    """
    rows: list[dict] = []
    current_date = None
    date_pattern = re.compile(r"^(\d{2}[A-Z]{3}\d{4})_")

    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # skip header
        except StopIteration:
            return pd.DataFrame()

        for row in reader:
            if not row:
                continue
            cell = row[0].strip()

            # Date header row
            m = date_pattern.match(cell)
            if m:
                try:
                    current_date = datetime.strptime(m.group(1), "%d%b%Y").date()
                except ValueError:
                    pass
                continue

            # Skip summary and blank
            if cell in ("Summary", "") or current_date is None:
                continue

            # Bar row: "HH:MM - HH:MM"
            if " - " not in cell:
                continue

            start_str = cell.split(" - ")[0].strip()
            try:
                t = datetime.strptime(start_str, "%H:%M").time()
            except ValueError:
                continue

            naive_dt = datetime.combine(current_date, t)
            try:
                dt_et = ET.localize(naive_dt, is_dst=None)
            except pytz.exceptions.AmbiguousTimeError:
                dt_et = ET.localize(naive_dt, is_dst=False)
            except pytz.exceptions.NonExistentTimeError:
                # Clock springs forward — skip the missing hour
                continue

            rows.append(
                {
                    "timestamp_et": dt_et,
                    "source": source,
                    "meeting_id": meeting_id,
                    "open": _safe_float(row[3]) if len(row) > 3 else None,
                    "high": _safe_float(row[4]) if len(row) > 4 else None,
                    "low": _safe_float(row[5]) if len(row) > 5 else None,
                    "close": _safe_float(row[1]) if len(row) > 1 else None,
                    "tick_count": _safe_float(row[6]) if len(row) > 6 else None,
                }
            )

    return pd.DataFrame(rows)


# Source → (subdirectory, filename prefix pattern)
_INTRADAY_SOURCES: dict[str, tuple[str, str]] = {
    # FX pairs
    "USDEUR": ("currency-data/USDEUR", "USDEUR_currency_data_{mid}.csv"),
    "USDJPY": ("currency-data/USDJPY", "USDJPY_currency_data_{mid}.csv"),
    "USDGBP": ("currency-data/USDGBP", "USDGBP_currency_data_{mid}.csv"),
    "USDCAD": ("currency-data/USDCAD", "USDCAD_currency_data_{mid}.csv"),
    # US Treasuries
    "UST2Y": ("treasury-data/2y_treasury", "USGG2YR_treasury_data_{mid}.csv"),
    "UST10Y": ("treasury-data/10y_treasury", "USGG10YR_treasury_data_{mid}.csv"),
    # Equity / vol indices
    "SPX": ("index-data/SPX", "SPX_index_data_{mid}.csv"),
    "VIX": ("index-data/VIX", "VIX_index_data_{mid}.csv"),
}


def load_intraday_bars(
    root: pathlib.Path,
    meeting_ids: list[str],
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Load all intraday 5-minute bar CSVs for the given meetings and sources.

    Parameters
    ----------
    root:         Project data_raw directory (e.g. Path("data-raw"))
    meeting_ids:  List of YYYYMMDD strings (from load_fomc_metadata)
    sources:      Subset of INTRADAY_SOURCES keys; default = all sources

    Returns a single DataFrame with canonical schema:
        timestamp_et  (tz-aware ET), source, meeting_id,
        open, high, low, close, tick_count
    """
    base = root / "bloomberg-exports"
    if sources is None:
        sources = list(_INTRADAY_SOURCES.keys())

    all_frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for source in sources:
        subdir, fname_tpl = _INTRADAY_SOURCES[source]
        src_dir = base / subdir
        for mid in meeting_ids:
            fpath = src_dir / fname_tpl.format(mid=mid)
            if not fpath.exists():
                missing.append(f"{source}/{mid}")
                continue
            df = _parse_intraday_csv(fpath, mid, source)
            if not df.empty:
                all_frames.append(df)

    if missing:
        logger.warning("Missing files (%d): %s", len(missing), missing[:10])

    if not all_frames:
        logger.warning("No intraday data loaded!")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["source", "meeting_id", "timestamp_et"]).reset_index(drop=True)
    logger.info(
        "Loaded %d bars across %d sources × %d meetings",
        len(combined),
        combined["source"].nunique(),
        combined["meeting_id"].nunique(),
    )
    return combined


# ── FOMC Transcripts ──────────────────────────────────────────────────────────


def _extract_pdf_text(fpath: pathlib.Path) -> str:
    """Extract full text from a PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError("pdfplumber is required: pip install pdfplumber") from e

    text_parts: list[str] = []
    with pdfplumber.open(fpath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def load_transcripts(root: pathlib.Path, meeting_ids: list[str]) -> dict[str, dict[str, str]]:
    """Extract text from all FOMC statement and press-conference PDFs.

    Returns a nested dict:
        {meeting_id: {"statement": <text>, "press_conf": <text>}}

    Missing PDFs result in an empty string for that key.
    """
    transcript_dir = root / "fomc-transcripts"
    stmt_dir = transcript_dir / "statements"
    pc_dir = transcript_dir / "press_conf"

    results: dict[str, dict[str, str]] = {}
    for mid in meeting_ids:
        stmt_path = stmt_dir / f"monetary{mid}a1.pdf"
        pc_path = pc_dir / f"FOMCpresconf{mid}.pdf"

        stmt_text = ""
        if stmt_path.exists():
            logger.debug("Extracting statement %s", stmt_path.name)
            stmt_text = _extract_pdf_text(stmt_path)
        else:
            logger.warning("Statement PDF missing: %s", stmt_path.name)

        pc_text = ""
        if pc_path.exists():
            logger.debug("Extracting press conf %s", pc_path.name)
            pc_text = _extract_pdf_text(pc_path)
        else:
            logger.warning("Press conf PDF missing: %s", pc_path.name)

        results[mid] = {"statement": stmt_text, "press_conf": pc_text}

    n_stmt = sum(1 for v in results.values() if v["statement"])
    n_pc = sum(1 for v in results.values() if v["press_conf"])
    logger.info("Transcripts: %d statements, %d press conferences loaded", n_stmt, n_pc)
    return results


def save_transcripts_json(transcripts: dict[str, dict[str, str]], out_path: pathlib.Path) -> None:
    """Serialize transcript dict to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    logger.info("Transcripts saved to %s", out_path)


def load_transcripts_json(path: pathlib.Path) -> dict[str, dict[str, str]]:
    """Load pre-extracted transcripts from JSON (fast path)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
