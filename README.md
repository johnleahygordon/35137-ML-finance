# 35137 ML in Finance

Code and documentation for 35137 (Machine Learning in Finance) at Chicago Booth.

## Repository Structure

```
35137-ML-finance/
├── HW1/                  Homework 1
├── HW2/                  Homework 2 — ML Portfolio Selection
├── HW3/                  Homework 3 (empty)
└── final-project/        Final research project
```

---

### HW1

Initial homework assignment using macroeconomic and market data.

```
HW1/
├── HW1_submission.ipynb   Main assignment notebook
├── hw1.pdf                Assignment specification
└── data/
    ├── FREDMD.csv         Federal Reserve macroeconomic dataset
    └── gw.csv             Market data
```

---

### HW2 — ML Portfolio Selection

Applies machine learning models (OLS, Ridge, LASSO, ElasticNet, Huber, GBT) to stock return prediction and portfolio construction. Q1 evaluates model performance on large/small-cap stocks; Q2 uses PCA on factor loadings for portfolio construction.

```
HW2/
├── configs/default.yaml   Train/test splits, ML hyperparameters, portfolio rules
├── data/
│   ├── largeml.pq         500 large-cap stocks (monthly returns + characteristics)
│   ├── smallml.pq         1000 small-cap stocks (same format)
│   └── lsret.csv          Long-short portfolio returns by characteristic
├── scripts/run_all.py     Single entry point for the full pipeline
├── src/                   Pipeline modules (config, data, ML models, portfolio, plotting, Q1/Q2 logic)
├── outputs/               Results: Sharpe ratio plots, predictions, comparison tables, report
├── hw2.pdf                Assignment specification
└── README.md              Setup and run instructions
```

---

### final-project — FX Reactions to FOMC Announcements

**Research question:** Does adding more sophisticated features (macro theory signals, cross-asset signals, LLM text scores, sentence embeddings) improve out-of-sample prediction of USD currency moves around FOMC announcements?

**Setup:** 41 FOMC meetings (Jan 2021 – Jan 2026), 4 USD pairs (EUR, JPY, GBP, CAD), 2 intraday windows (statement: 14:00–14:30 ET; digestion: 14:30–16:00 ET) → 328 panel observations. Models are evaluated with Leave-One-Meeting-Out cross-validation across a 4-rung model ladder.

**Key finding:** Text features (LLM rubric scores, sentence embeddings) consistently hurt both RMSE and directional accuracy. Best performer is LASSO (~10 bps RMSE, statement window).

```
final-project/
├── configs/config.yaml        Master config: meeting dates, windows, feature grids, hyperparameters
├── data-raw/                  Source data (not pipeline-generated)
│   ├── fomc-transcripts/      82 FOMC PDFs (41 statements + 41 press conference transcripts)
│   └── bloomberg-exports/     5-min OHLCV bars (4 FX pairs, SPX, VIX, 2Y/10Y treasuries),
│                              daily policy rates (Fed, BOC, BOE, BOJ, ECB), FOMC schedule
├── data-clean/                Pipeline output parquets
│   ├── fomc_metadata.parquet      Meeting details, timestamps, policy rates
│   ├── policy_rates.parquet       Policy rate time series (all central banks)
│   ├── intraday_bars.parquet      5-minute OHLCV for all pairs/indices
│   ├── targets.parquet            Log-returns for statement and digestion windows
│   ├── features_structured.parquet  Rung 2–3 features (theory, cross-asset signals)
│   ├── features_text.parquet      Rung 4 features (keywords, LLM scores, embeddings)
│   ├── panel_final.parquet        Complete 328-row panel for modeling
│   ├── embeddings_cache.parquet   Cached sentence embeddings
│   ├── llm_scores_cache.parquet   Cached LLM rubric responses
│   └── transcripts.json           Parsed statement + press conference text
├── notebooks/                 Exploratory analysis
│   ├── 01_eda_raw_data.ipynb      Raw data inspection, coverage heatmap
│   ├── 02_eda_targets.ipynb       Return distributions, predict-zero baseline
│   ├── 03_eda_features.ipynb      Structured features, correlation matrix
│   ├── 04_text_scoring.ipynb      Keyword, LLM, and embedding scores
│   └── 05_model_results.ipynb     Full model ladder, diagnostics, P&L
├── outputs/
│   ├── model_results.csv          Master metrics table (MAE, RMSE, dir_acc, OOS R²)
│   ├── figures/                   12 PNG plots (RMSE/dir_acc ladders, P&L, PCA, residuals)
│   └── prompts/rubric_v1.txt      LLM prompt for 7-dimensional rubric scoring
├── scripts/run_pipeline.py    End-to-end orchestration (idempotent; use --force to recompute)
├── src/                       Pipeline modules
│   ├── config.py                  YAML → dataclass config loader
│   ├── ingest.py                  Raw data parsing (CSV, PDF text extraction)
│   ├── clean.py                   Schema enforcement, QA, parquet I/O
│   ├── targets.py                 Log-return computation, panel construction
│   ├── features_structured.py     Rungs 1–3: theory-based + cross-asset features
│   ├── features_text.py           Rung 4: keyword scoring, LLM rubric, embeddings + PCA
│   ├── models.py                  Model definitions with LOMO cross-validation
│   ├── eval.py                    Metrics, panel assembly, P&L simulation
│   └── plots.py                   Plotting utilities
├── requirements.txt           Python dependencies
└── README.md                  Project-level setup, methods, and results
```

#### Running the final-project pipeline

```bash
cd final-project
set -a && source ../secrets.env && set +a
python scripts/run_pipeline.py              # idempotent (skips existing parquets)
python scripts/run_pipeline.py --force     # recompute everything
python scripts/run_pipeline.py --skip-llm --skip-embed  # skip API/embedding steps
```
