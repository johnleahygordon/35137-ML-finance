# HW2 — ML in Finance

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib pyarrow statsmodels joblib pyyaml
```

## Data

Place the following files in `data/`:
- `largeml.pq` — 500 large-cap stocks (monthly returns + characteristics)
- `smallml.pq` — 1000 small-cap stocks (same format)
- `lsret.csv` — long-short portfolio returns for each characteristic

## Run

```bash
cd HW2
python scripts/run_all.py
```

All outputs are saved to `outputs/`.

## Project Structure

```
HW2/
├── configs/
│   └── default.yaml          # All parameters (splits, grids, paths)
├── data/                     # Raw data (not committed)
├── outputs/                  # Generated artifacts
├── scripts/
│   └── run_all.py            # Single entry point
├── src/
│   ├── config.py             # Configuration loader
│   ├── data.py               # Data loading, schema inference, time splits
│   ├── ml_models.py          # ML model builders and grid search
│   ├── plotting.py           # Plotting utilities
│   ├── portfolio.py          # Portfolio construction and metrics
│   ├── question1.py          # Q1(a)–(e) implementation
│   └── question2.py          # Q2(a)–(e) implementation
└── README.md
```

## Configuration

Edit `configs/default.yaml` to change:
- Data paths
- Train/validation/test split rules (20y / 12y / remainder)
- Portfolio construction (decile long-short)
- ML hyperparameter grids
- Q2 indicator cutoff year (2004)
