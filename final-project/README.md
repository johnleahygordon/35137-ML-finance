# Final Project

## Overview

_[Placeholder: brief description of the project, research question, and methods.]_

---

## Data

All raw data lives under `data/`. There are two main sources: FOMC text documents and Bloomberg market data exports.

### FOMC Transcripts (`data/fomc-transcripts/`)

PDF documents sourced from the Federal Reserve website, one file per FOMC meeting.

| Subfolder | Contents | Naming convention |
|-----------|----------|-------------------|
| `statements/` | Official post-meeting monetary policy statements | `monetary{YYYYMMDD}a1.pdf` |
| `press_conf/` | Chair press conference transcripts | `FOMCpresconf{YYYYMMDD}.pdf` |

Coverage: 41 meetings, January 2021 – January 2026.

---

### Bloomberg Exports (`data/bloomberg-exports/`)

Market data exported from Bloomberg Terminal. Intraday files are keyed by FOMC meeting date so they can be matched directly to the text documents above.

#### FOMC Meeting Schedule (`fomc-data/fomc_meetings.csv`)

One row per meeting. Columns: `announcement_date`, `announcement_time`, `lower_rate`, `upper_rate`, `votes_for`, `votes_against`, `policy_concern`.

#### Currency Data (`currency-data/`)

Intraday 5-minute OHLCV bars for four USD FX pairs, one CSV per FOMC meeting day.

| Subfolder | Pair | Coverage |
|-----------|------|----------|
| `USDCAD/` | USD/CAD | Jan 2021 – Jan 2026 |
| `USDEUR/` | USD/EUR | Jan 2008 – Jan 2026 |
| `USDGBP/` | USD/GBP | Jan 2021 – Jan 2026 |
| `USDJPY/` | USD/JPY | Jan 2021 – Jan 2026 |

Columns per file: `Time Interval`, `Close`, `Net Chg`, `Open`, `High`, `Low`, `Tick Count`, `Volume`. The first data row is a daily summary, followed by 5-minute bars.

#### Equity & Volatility Index Data (`index-data/`)

Intraday 5-minute OHLCV bars, one CSV per FOMC meeting day, same column schema as currency data.

| Subfolder | Instrument | Coverage |
|-----------|------------|----------|
| `SPX/` | S&P 500 Index | Jan 2021 – Jan 2026 |
| `VIX/` | CBOE Volatility Index | Jan 2021 – Jan 2026 |

#### Policy Rate Time Series (`policy-rate-data/`)

Daily closing policy rate levels for five central banks, 2021–2026. One CSV per central bank.

| File | Central bank | Bloomberg ticker |
|------|-------------|-----------------|
| `Fed_policy_rate_FEDL01_2021-26.csv` | Federal Reserve | FEDL01 |
| `BOC_policy_rate_OVERNIGHT_2021-26.csv` | Bank of Canada | OVERNIGHT |
| `BOE_policy_rate_UKBRBASE_2021-26.csv` | Bank of England | UKBRBASE |
| `BOJ_policy_rate_BOJEPR_2021-26.csv` | Bank of Japan | BOJEPR |
| `ECB_policy_rate_EURR002W_2021-26.csv` | European Central Bank | EURR002W |

Columns: `Date`, `PX_LAST`, `Change`, `% Change`, `PX_BID`.

#### US Treasury Yield Data (`treasury-data/`)

Intraday 5-minute yield data for two benchmark maturities, one CSV per FOMC meeting day.

| Subfolder | Instrument | Bloomberg ticker |
|-----------|------------|-----------------|
| `2y_treasury/` | 2-year US Treasury yield | USGG2YR |
| `10y_treasury/` | 10-year US Treasury yield | USGG10YR |

---

## Project Structure

```
final-project/
├── data/
│   ├── fomc-transcripts/       # FOMC statement and press conference PDFs
│   └── bloomberg-exports/      # Market data exports (see above)
├── configs/                    # [Placeholder: config files]
├── scripts/                    # [Placeholder: data download / preprocessing scripts]
├── src/                        # [Placeholder: source modules]
└── outputs/                    # [Placeholder: results, figures, model artifacts]
```

---

## Setup

_[Placeholder: environment setup, dependencies, reproduction instructions.]_

---

## Methods

_[Placeholder: feature engineering, model architecture, training procedure, evaluation.]_

---

## Results

_[Placeholder: key findings, tables, figures.]_

---

## References

_[Placeholder: papers, data sources, related work.]_
