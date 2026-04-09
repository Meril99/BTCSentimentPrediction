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
| `XGBOOST_ADABOOST.ipynb` | XGBoost and AdaBoost — apparent accuracy improvement over RF (random split) |
| `robust_evaluation.ipynb` | Time-based validation, lag features, walk-forward CV, bootstrap CIs, McNemar's test |

## Results

With random train/test splits, AdaBoost reaches 73% and XGBoost 67%. However, **random splits on time-series data cause data leakage** — the model trains on future data and "predicts" the past.

With proper chronological validation (train on 2021–2022, test on 2023), all models collapse to baseline (~50–53%). Bootstrap confidence intervals confirm none are statistically significant.

| Model | Random Split | Time-Based Split | Inflation |
|-------|-------------|-----------------|-----------|
| AdaBoost | 73.37% | 49.40% | +23.97% |
| XGBoost | 67.46% | 52.98% | +14.48% |
| Random Forest | 56.80% | 51.19% | +5.61% |
| Baseline (majority) | — | 51.43% | — |

## Key Findings

- Tweet sentiment does not predict BTC daily price direction under proper temporal validation
- Granger causality and lag correlation tests confirm no causal relationship in either direction
- The sentiment-price correlation reverses across bull/bear regimes, cancelling out in aggregate
- Market features (Open, Volume, Volatility) outperform sentiment features with Random Forest
- More flexible models exploit data leakage more aggressively (AdaBoost +24% vs RF +6%)
- Institutional actors (governments, exchanges) dominate price action over retail Twitter sentiment

## Why It Fails — and What Might Not

The naive approach of averaging daily tweet sentiment into a single score and predicting next-day direction doesn't work. But the EDA reveals *why*, which points to what a better approach would look like:

- **Daily aggregation is too coarse** — a viral tweet from a major figure moves BTC within minutes. By the next day, the market has already priced it in. Intraday granularity could capture signal that daily averages destroy.
- **Not all tweets are equal** — the May 2021 crash analysis shows institutional announcements (US government selling 41,500 BTC, Binance regulatory pressure) moved the market despite overwhelmingly positive retail sentiment. Monitoring *specific high-impact actors* rather than averaging all tweets could work.
- **Polarity is too crude** — Word2Vec embeddings (semantic content) added more signal than VADER's positive/negative scores. "Government sells 41,500 BTC" and "Bitcoin is the future" both matter, but for completely different reasons that a polarity score can't distinguish.

## Tech Stack

Python 3.11 | pandas | scikit-learn | XGBoost | VADER | Word2Vec | statsmodels | matplotlib | seaborn
