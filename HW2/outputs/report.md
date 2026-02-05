# HW2 — ML in Finance: Results Report

## Question 1

### 1(a) Characteristic Portfolios — Large Caps

**206 characteristics** evaluated (3 dropped due to >80% missing).

- **Best**: High52 (Sharpe = 4.53) — 52-week high ratio, a momentum-related signal
- **Worst**: ReturnSkew3F (Sharpe = -4.12) — return skewness factor, investors may pay a premium for positive skew

The top performers include momentum signals (High52, Mom12m, MomSeason variants) and profitability measures. The worst performers include measures of volatility and skewness, consistent with the literature on low-risk anomalies.

See `q1a_large_sharpe_plot.png` and `q1a_large_sharpe_by_char.csv`.

### 1(b) ML Models — Large Caps

**Time split**: train ≤1945 (942 obs), val 1946–1957 (715 obs), test >1957 (77,140 obs).

| Model | OOS R² |
|---|---|
| ElasticNet | 0.179 |
| Lasso | 0.148 |
| GBR | 0.146 |
| RBF+Lasso | 0.016 |
| RBF+ElasticNet | 0.016 |
| RBF+Ridge | 0.015 |
| PLS_RBF | -0.048 |
| PLS_linear | -27.6 |
| Ridge | -436.4 |
| OLS | -1,488.2 |

**Interpretation**: Regularised linear models (Lasso, ElasticNet) perform best, as they effectively select the most predictive characteristics while shrinking noise. OLS and Ridge severely overfit given the small training set (942 obs, 206 features) — a classic p ≫ n problem. The RBF expansion adds dimensionality without sufficient regularisation benefit, while gradient boosting captures modest non-linearities. The very large test period (65 years) provides a robust evaluation window.

See `q1b_large_oos_r2_by_model.csv` and `q1b_large_hyperparams.json`.

### 1(c) ML Portfolio Sharpes — Large Caps

| Model | Annualised Sharpe |
|---|---|
| ElasticNet | 5.45 |
| Lasso | 5.06 |
| GBR | 4.84 |
| PLS_linear | 4.78 |
| OLS | 2.14 |
| Ridge | 2.12 |
| PLS_RBF | 0.06 |
| RBF+Ridge | -0.07 |

- **Best ML model (ElasticNet, 5.45)** outperforms the best individual characteristic (High52, 4.53) by combining information from multiple signals.
- Even OLS achieves a reasonable Sharpe (2.14) despite negative OOS R², because signal ranking for portfolio formation is more robust than point predictions.

See `q1c_large_ml_sharpe_comparison.csv` and `q1c_large_ml_sharpe_plot.png`.

### 1(d) Small Caps

**143 characteristics** evaluated (66 dropped due to >80% missing — many characteristics unavailable for small caps in early decades).

| | Large | Small |
|---|---|---|
| Best char | High52 (4.53) | High52 (2.72) |
| Worst char | ReturnSkew3F (-4.12) | ReturnSkew (-2.60) |
| Best ML R² | ElasticNet (0.179) | GBR (0.183) |
| Best ML Sharpe | ElasticNet (5.45) | GBR (2.98) |

**Large vs Small comparison**: Large-cap characteristic Sharpes are generally higher, consistent with more liquid markets and more efficient price discovery making signals more persistent. Small-cap ML models achieve comparable OOS R² but lower Sharpes, likely due to higher idiosyncratic volatility diluting the signal-to-noise ratio at the portfolio level. GBR performs relatively better for small caps, suggesting more non-linear relationships in less-followed stocks.

### 1(e) Max Sharpe Attempt

| | Large | Small |
|---|---|---|
| Sharpe | 5.57 | 2.09 |

**Strategy**: Ensemble of all ML model z-scored predictions, z-score weighted portfolio, volatility scaled to 15% annualised target vol. The ensemble benefits from diversification across model architectures (linear vs tree-based vs PLS).

---

## Question 2

### 2(a) PCA Latent Factors

| Portfolio Set | 80% variance | 90% variance | 95% variance |
|---|---|---|---|
| lsret | 54 factors | 90 factors | 120 factors |
| Large char | 64 factors | 96 factors | 124 factors |
| Small char | 37 factors | 59 factors | 79 factors |

**Interpretation**: All three portfolio sets require many factors (50+) to explain 80% of variation, indicating that cross-sectional return variation is genuinely high-dimensional. The small-cap set requires fewer factors, consistent with fewer underlying characteristics and potentially stronger common factor structure. The lsret portfolios (pre-constructed long-short returns from openassetpricing) have a richer factor structure than the rank-sort portfolios we construct.

See `q2a_explained_variance_*.png` and `q2a_factor_loadings_*.csv`.

### 2(b) Indicator Prediction (pre-2004)

The Britten-Jones indicator approach (regress 1 on portfolio returns, no intercept) yields regularised mean-variance optimal weights.

| Dataset | Ridge Sharpe | Lasso Sharpe |
|---|---|---|
| lsret | 2.23 (α=1000) | 2.07 (α=0.1) |
| Large char | 9.62 (α=1) | 8.70 (α=0.01) |
| Small char | 2.91 (α=100) | 3.00 (α=0.1) |

**Interpretation**: The large-cap characteristic portfolios achieve remarkably high Sharpes via the indicator approach, reflecting that the mean-variance optimiser can combine the best characteristics effectively. Ridge tends to outperform Lasso for lsret (many non-zero weights), while results are mixed for characteristic portfolios.

### 2(c) Sharpe vs # PCA Factors

| Dataset | Best Ridge Sharpe |
|---|---|
| lsret | 0.81 |
| Large char | 1.13 |
| Small char | 3.65 |

**Interpretation**: PCA compression before indicator regression generally reduces Sharpe relative to using the full portfolio set (Q2b). This suggests that the information relevant for mean-variance optimisation is spread across many components and cannot be efficiently captured by a low-rank approximation. The small-cap set is an exception, where a few PCA factors capture the essential information.

See `q2c_sharpe_vs_k_*.png` and `q2c_sharpe_vs_k_*.csv`.

### 2(d) Large/Small Comparison

All three dataset results are compiled in `q2d_comparison.csv`. Key findings:

- The full-dimensional indicator approach (Q2b) generally outperforms PCA-compressed versions (Q2c)
- Large-cap characteristic portfolios yield the highest Sharpes via the indicator method
- Lasso selects fewer portfolios but performance is comparable to ridge
- Small-cap portfolios benefit more from PCA compression than large-cap

### 2(e) Max Sharpe Attempt (lsret)

**Best configuration**: PCA with k=1 factor + Ridge (α=0.01), vol-scaled

| Metric | Value |
|---|---|
| Raw Sharpe | 0.29 |
| Vol-scaled Sharpe | 0.30 |

**Strategy**: Searched over (k, α) pairs, selecting on validation Sharpe, then applied volatility scaling. The relatively low OOS Sharpe for lsret compared to Q2b suggests that the full-dimensional approach is preferable for the lsret dataset, and the Q2c-style compression loses too much information.

---

## Assumptions

- Returns column (`ret`) contains string 'C' for some rows → treated as missing (349 in large, 464 in small).
- Missing characteristics imputed with cross-sectional median per month.
- Features with >80% missing values excluded from ML models.
- OOS R² computed against a zero-mean benchmark (SS_tot = Σy²).
- Portfolio: equal-weight decile long-short (top decile − bottom decile).
- Time splits: first 20 unique calendar years = train, next 12 = validation, remainder = test.
- Q2 indicator cutoff: all data before 2004 for training/validation.
- Q2 indicator regression uses `fit_intercept=False` (Britten-Jones approach).
- Volatility scaling targets 15% annualised vol with leverage capped at [0.5, 3.0].
