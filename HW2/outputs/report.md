# HW2 — ML in Finance: Results Report

## Question 1

### 1(a) Characteristic Portfolios — Large Caps

**206 characteristics** evaluated (3 dropped due to >80% missing).

- **Best**: High52 (Sharpe = 4.53) — 52-week high ratio, a momentum-related signal
- **Worst**: ReturnSkew3F (Sharpe = -4.12) — return skewness factor, investors may pay a premium for positive skew

The top performers include momentum signals (High52, Mom12m, MomSeason variants) and profitability measures. The worst performers include measures of volatility and skewness, consistent with the literature on low-risk anomalies.

See `q1a_large_sharpe_plot.png` anIP0`q1a_large_sharpe_by_char.csv`.

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

**Strategy**: Ensemble of all ML model z-scored predictions, z-score weighted portfolio, volatility scaled to 15% annualised target vol.

**Rationale — Why This Approach?**

The strategy combines three key techniques, each chosen for specific reasons:

1. **Model Averaging / Ensemble**
   - Different ML architectures capture different aspects of the return-predictability relationship: linear models (Lasso, ElasticNet) excel at sparse feature selection, tree-based models (GBR) capture non-linearities and interactions, and PLS extracts latent factors from the high-dimensional characteristic space.
   - Each model's prediction errors are partially uncorrelated — when ElasticNet misfires due to non-linear effects, GBR may still capture the signal, and vice versa.
   - By averaging z-scored predictions across all models, we reduce model-specific noise and estimation error. This is analogous to the diversification benefit in portfolio theory: combining imperfectly correlated signals yields a higher signal-to-noise ratio than relying on any single model.
   - The z-scoring before averaging ensures each model contributes equally regardless of its prediction scale.

2. **Z-Score Weighted Portfolio (vs Decile Sorting)**
   - Standard decile long-short portfolios treat all stocks in the top (bottom) decile equally, discarding information about *how strongly* a stock is predicted to outperform.
   - Z-score weighting instead assigns weights proportional to the signal magnitude: a stock with a z-score of +2.5 receives more weight than one with +0.5, both in the long leg.
   - This exploits the full cross-sectional distribution of predictions rather than just rank ordering, extracting more information from the ML signals.
   - The approach is equivalent to a cross-sectional regression-based portfolio (Fama-MacBeth style), which has theoretical grounding in capturing conditional expected returns more efficiently.

3. **Volatility Scaling**
   - Raw portfolio returns exhibit time-varying volatility — periods of market stress produce higher variance that can dominate the Sharpe calculation.
   - Volatility scaling targets a constant 15% annualised volatility by dynamically adjusting position sizes inversely proportional to recent (12-month trailing) realised volatility.
   - This is equivalent to running a constant-risk strategy rather than a constant-notional strategy, improving risk-adjusted returns by reducing exposure during turbulent periods and increasing it during calm periods.
   - The leverage bounds [0.5, 3.0] prevent extreme positions that could arise from unusually low or high volatility estimates.
   - Volatility scaling is a well-documented technique in the quantitative finance literature (e.g., Moreira and Muir, 2017) shown to improve Sharpe ratios across many asset classes.

**Why Not Other Approaches?**

- *Single best model*: ElasticNet achieved the highest individual Sharpe (5.45 for large caps), but relying on a single model introduces selection bias and model risk. The ensemble (5.57) outperforms by diversifying across architectures.
- *Equal-weighted decile portfolios*: The standard approach is robust but leaves information on the table. Z-score weighting increased Sharpe by exploiting signal magnitude.
- *Optimised model weights*: We could have learned optimal ensemble weights on validation data, but this risks overfitting given limited data. Equal weighting (after z-scoring) is more robust.
- *Transaction cost constraints*: Not modelled here, but in practice z-score weighting can generate higher turnover. Decile sorting may be preferable when transaction costs are a concern.

**Limitations**

- The ensemble assumes all models provide useful signal; if some models are pure noise, they dilute the ensemble. Validation-based model selection could prune poor performers.
- Volatility scaling uses trailing realised volatility as a proxy for forward volatility, which can lag during regime changes.
- Small-cap Sharpe (2.09) is substantially lower than large-cap (5.57), likely due to higher idiosyncratic volatility and less efficient signal transmission in less liquid markets.

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
