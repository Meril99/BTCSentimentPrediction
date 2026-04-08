# BTC Sentiment-Price Prediction

Predicting daily Bitcoin price direction (up/down) by combining Twitter sentiment analysis with market data. Built as a data mining project exploring whether social media signals carry predictive power for crypto price movements.

## Pipeline

```
Raw tweets (7.4M) + BTC OHLCV ──► Preprocessing ──► Feature Engineering ──► Models ──► Evaluation
```

**Data**: 842 daily samples (2021–2023), each combining tweet sentiment (VADER compound + pos/neg/neu scores), engagement metrics (favorites, retweets, replies), custom importance coefficient, and BTC market data (OHLC, volume).

## Notebooks

| Notebook | Description |
|----------|-------------|
| `preprocessing.ipynb` | Text cleaning, VADER sentiment scoring, Word2Vec vectorization, feature construction |
| `general_ExplanatoryDataAnalysis.ipynb` | Initial EDA — distributions, correlations, word clouds, crash/surge analysis |
| `advanced_eda.ipynb` | Stationarity tests (ADF), lag correlation analysis, Granger causality, market regime detection |
| `randomForestModel.ipynb` | Random Forest with multiple feature subsets (baseline, +Word2Vec, market-only, tweet-only) |
| `XGBOOST_ADABOOST.ipynb` | XGBoost and AdaBoost — significant accuracy improvement over RF |
| `robust_evaluation.ipynb` | Time-based validation, lag features, walk-forward CV, bootstrap CIs, McNemar's test |

## Results

Standard tree-based models (Random Forest) plateau around 50–56% accuracy — barely above the majority-class baseline. **XGBoost and AdaBoost break through** to 67–73% by better handling feature interactions and weak signals.

The `robust_evaluation` notebook addresses data leakage concerns (random vs time-based splits), engineers lag features to prevent look-ahead bias, and reports bootstrap confidence intervals for all metrics.

## Key Findings

- Market features alone (Open, Volume, Volatility) outperform tweet-only features with Random Forest
- Combining sentiment + market data with boosted models yields the best results
- AdaBoost with shallow decision trees achieves the highest test accuracy
- Lag analysis reveals whether sentiment leads or follows price movements

## Tech Stack

Python 3.11 | pandas | scikit-learn | XGBoost | VADER | Word2Vec | statsmodels | matplotlib | seaborn
