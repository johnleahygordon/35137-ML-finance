# FX Moves Around FOMC Meetings — Model Ladder

## Overview

**Research question:** Are "smarter" models actually better at predicting USD FX moves on Federal Reserve decision days?

This project evaluates whether increasing model complexity improves prediction of intraday FX returns around FOMC announcements. Using 41 meetings (January 2021 – January 2026) across four USD currency pairs (EUR, JPY, GBP, CAD), we run a four-rung **model ladder**:

| Rung | Models | Features |
|------|--------|----------|
| 1 | Predict-zero, Historical mean | None (baselines) |
| 2 | OLS | Policy rate differentials, rate change |
| 3 | Ridge, LASSO, ElasticNet, Huber, GBT | Structured: theory + pre-announcement cross-asset signals |
| 4 | Rung-3 + text | Keyword scores, LLM rubric (Claude), sentence embeddings + PCA |

**Prediction targets:** Log-returns in two windows relative to the 2:00 PM ET announcement:
- **Statement window** (2:00–2:30 PM ET): immediate reaction to the decision and statement
- **Digestion window** (2:30–4:00 PM ET): adaptation through the press conference

Sign convention: positive = USD appreciation.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set Claude API key (required for LLM scoring — Stage 4b)
export ANTHROPIC_API_KEY="sk-ant-..."

# Run the full pipeline
python scripts/run_pipeline.py

# Skip API calls (use cached results if available)
python scripts/run_pipeline.py --skip-llm --skip-embed

# Force recompute all steps
python scripts/run_pipeline.py --force
```

**Python 3.10+ required.** Run commands from the `final-project/` directory.

### EDA Notebooks (run in order)

```
notebooks/01_eda_raw_data.ipynb    # Raw data inspection, coverage heatmap
notebooks/02_eda_targets.ipynb     # Return distributions, predict-zero baseline
notebooks/03_eda_features.ipynb    # Structured features, correlation matrix
notebooks/04_text_scoring.ipynb    # Keyword + LLM + embedding scores
notebooks/05_model_results.ipynb   # Full ladder, diagnostics, P&L
```

---

## Project Structure

```
final-project/
├── configs/
│   └── config.yaml                  # All parameters: windows, pairs, hypergrids, LLM settings
├── requirements.txt
├── notebooks/
│   ├── 01_eda_raw_data.ipynb
│   ├── 02_eda_targets.ipynb
│   ├── 03_eda_features.ipynb
│   ├── 04_text_scoring.ipynb
│   └── 05_model_results.ipynb
├── src/
│   ├── config.py                    # Dataclass config loader
│   ├── ingest.py                    # Raw data loading (CSV → DataFrame, PDF → text)
│   ├── clean.py                     # Canonical schema, QA, parquet I/O
│   ├── targets.py                   # Log-return computation, panel construction
│   ├── features_structured.py       # Rungs 1–3: theory + cross-asset features
│   ├── features_text.py             # Rung 4: keyword scoring, LLM rubric, embeddings+PCA
│   ├── models.py                    # Model definitions + LOO-CV
│   ├── eval.py                      # Metrics, panel assembly, P&L check
│   └── plots.py                     # Reusable plotting utilities
├── scripts/
│   └── run_pipeline.py              # End-to-end orchestration
├── data-raw/
│   ├── fomc-transcripts/            # FOMC statement and press conference PDFs
│   └── bloomberg-exports/           # Market data exports (see Data section below)
├── data-clean/                      # Parquet files written by the pipeline
│   ├── fomc_metadata.parquet
│   ├── policy_rates.parquet
│   ├── intraday_bars.parquet
│   ├── targets.parquet
│   ├── features_structured.parquet
│   ├── features_text.parquet
│   └── panel_final.parquet
└── outputs/
    ├── model_results.csv            # Metrics table (MAE, RMSE, dir_acc, OOS R²)
    ├── figures/                     # All PNG plots
    └── prompts/                     # LLM prompt templates (version-controlled)
        └── rubric_v1.txt
```

---

## Data

All raw data lives under `data-raw/`. Two main sources: FOMC text documents and Bloomberg market data.

### FOMC Transcripts (`data-raw/fomc-transcripts/`)

PDF documents sourced from the Federal Reserve website, one file per FOMC meeting.

| Subfolder | Contents | Naming convention |
|-----------|----------|-------------------|
| `statements/` | Official post-meeting monetary policy statements | `monetary{YYYYMMDD}a1.pdf` |
| `press_conf/` | Chair press conference transcripts | `FOMCpresconf{YYYYMMDD}.pdf` |

Coverage: 41 meetings, January 2021 – January 2026.

---

### Bloomberg Exports (`data-raw/bloomberg-exports/`)

Market data exported from Bloomberg Terminal. Intraday files are keyed by FOMC meeting date (YYYYMMDD). All timestamps in the CSV files are **Eastern Time**.

#### FOMC Meeting Schedule (`fomc_meetings.csv`)

One row per meeting. Columns: `announcement_date`, `announcement_time`, `lower_rate`, `upper_rate`, `votes_for`, `votes_against`, `policy_concern`.

#### Currency Data (`currency-data/`)

Intraday 5-minute OHLCV bars, one CSV per FOMC meeting day.

| Subfolder | Pair | Coverage |
|-----------|------|----------|
| `USDCAD/` | USD/CAD | Jan 2021 – Jan 2026 |
| `USDEUR/` | USD/EUR | Jan 2008 – Jan 2026 |
| `USDGBP/` | USD/GBP | Jan 2021 – Jan 2026 |
| `USDJPY/` | USD/JPY | Jan 2021 – Jan 2026 |

Quote convention: USDEUR = units of EUR per USD (positive move = USD appreciation).

Columns: `Time Interval`, `Close`, `Net Chg`, `Open`, `High`, `Low`, `Tick Count`, `Volume`. Each file spans two calendar days (overnight + announcement day).

#### Equity & Volatility Index Data (`index-data/`)

Same column schema as currency data. One CSV per FOMC meeting day.

| Subfolder | Instrument | Coverage |
|-----------|------------|----------|
| `SPX/` | S&P 500 Index | Jan 2021 – Jan 2026 |
| `VIX/` | CBOE Volatility Index | Jan 2021 – Jan 2026 |

#### Policy Rate Time Series (`policy-rate-data/`)

Daily closing policy rate levels for five central banks, 2021–2026.

| File | Central bank | Bloomberg ticker |
|------|-------------|-----------------|
| `Fed_policy_rate_FEDL01_2021-26.csv` | Federal Reserve | FEDL01 |
| `BOC_policy_rate_OVERNIGHT_2021-26.csv` | Bank of Canada | OVERNIGHT |
| `BOE_policy_rate_UKBRBASE_2021-26.csv` | Bank of England | UKBRBASE |
| `BOJ_policy_rate_BOJEPR_2021-26.csv` | Bank of Japan | BOJEPR |
| `ECB_policy_rate_EURR002W_2021-26.csv` | European Central Bank | EURR002W |

#### US Treasury Yield Data (`treasury-data/`)

Intraday 5-minute yield data, one CSV per FOMC meeting day.

| Subfolder | Instrument | Bloomberg ticker |
|-----------|------------|-----------------|
| `2y_treasury/` | 2-year US Treasury yield | USGG2YR |
| `10y_treasury/` | 10-year US Treasury yield | USGG10YR |

---

## Methods

### Target Construction

For each (meeting × pair × window), the log-return is computed as:

```
log_ret = ln(close_exit / close_entry)
```

where `close_entry` is the last 5-minute bar close at or before the window start, and `close_exit` is the last bar close within the window. Maximum panel size: 41 × 4 × 2 = 328 rows.

### Feature Engineering

**No-leakage guarantee:** all features for the statement window use data from before 14:00 ET; all features for the digestion window use the same pre-announcement data plus the statement text.

**Rung 2 — Theory:**
- `rate_change_bps`: Fed funds rate change at this meeting
- `fed_minus_foreign_pre`: Policy spread vs foreign CB (day before meeting)
- `spread_change`: Change in spread vs prior meeting

**Rung 3 — Cross-asset (pre-window: 10:00–13:55 ET):**
- 2Y and 10Y Treasury yield changes
- Yield curve slope change
- S&P 500 pre-announcement log-return
- VIX level and change

**Rung 4 — Text signals:**
- Keyword hawkish/dovish net score (deterministic)
- LLM rubric: 7 dimensions scored by `claude-opus-4-6` using `outputs/prompts/rubric_v1.txt`
- Sentence embeddings (all-MiniLM-L6-v2) + PCA reduction to 5–20 components

### Evaluation

- **Cross-validation:** Leave-One-Meeting-Out (LOMO) — train on 40 meetings, predict 1, grouped by `meeting_id` to prevent leakage
- **Metrics:** MAE, RMSE (in bps), directional accuracy (%), OOS R² vs predict-zero
- **Robustness:** outlier-drop (top 3 extreme meetings), classification vs regression, per-pair vs pooled

---

## Results

_Run `notebooks/05_model_results.ipynb` or `python scripts/run_pipeline.py` to generate results._

Key outputs:
- `outputs/model_results.csv` — full metric table
- `outputs/figures/ladder_rmse_statement.png` — main ladder chart

---

## References

- BIS Triennial Central Bank Survey (2025) — FX market turnover
- Federal Reserve FOMC materials: [federalreserve.gov/monetarypolicy/fomccalendars.htm](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- Bloomberg Terminal data (accessed via UChicago Booth)
