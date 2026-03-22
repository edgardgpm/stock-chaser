# StockChaser
#### Video Demo: <URL HERE - PENDING>

#### Description:
StockChaser is a Python project that employs a machine learning pipeline to predict whether a stock's closing price will be higher in N amount of days, using historical market data and technical indicators. The goal of the project is to build a clean, reproducible pipeline for financial prediction while following good ML and software engineering practices.

---

## Requirements
- Python 3.13.3
- Dependencies listed in `requirements.txt`

---

## Setup

1. Clone or download the project.
2. Install dependencies:
```
pip install -r requirements.txt
```

---

## How to Run

```
python main.py --symbol AAPL --start 2022-01-01 --end 2026-03-01 --threshold 0.25
```

All arguments are optional and fall back to the defaults defined in `config.py`.

---

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--symbol` | Stock ticker symbol to analyze (e.g. AAPL, MSFT, TSLA) | `AAPL` |
| `--start` | Start date for historical data in YYYY-MM-DD format | `2022-01-01` |
| `--end` | End date for historical data in YYYY-MM-DD format | `2026-03-01` |
| `--threshold` | Probability cutoff for classifying a prediction as UP. A lower value (e.g. 0.25) predicts UP more aggressively; a higher value (e.g. 0.75) requires more confidence before predicting UP. | `0.25` |

---

## How It Works

1. **Data Load** — Downloads historical OHLCV data for the given ticker via Yahoo Finance
2. **Feature Engineering** — Computes technical indicators: moving averages, multi-period returns, and rolling volatility
3. **Dataset Preparation** — Builds the feature matrix (X) and target vector (y), where y = 1 if price is higher in N days, else 0
4. **Time Split** — Splits data chronologically (no shuffling) to prevent data leakage
5. **Training** — Trains a Random Forest classifier on the training portion
6. **Evaluation** — Measures accuracy against a naive baseline and generates a confusion matrix
7. **Reporting** — Prints metrics, feature importances, and sample predictions to the console
8. **Outputs** — Saves predictions, metrics, and charts to the results folder

---

## Features
- Moving averages (5-day, 10-day)
- Multi-period returns (1-day, 2-day, 5-day)
- Rolling volatility (5-day)
- Automatic historical data download via Yahoo Finance
- Time-series safe train/test split
- Configurable probability threshold
- Model evaluation & artifact generation

---

## Model
Random Forest Classifier (`scikit-learn`), trained with `class_weight="balanced"` to handle imbalanced UP/DOWN class distributions.

---

## Outputs

Results are saved to the `stockchaser_results/` folder:

| File | Description |
|---|---|
| `{SYMBOL}_predictions.csv` | Feature values alongside actual and predicted labels |
| `{SYMBOL}_metrics.txt` | Baseline accuracy and model accuracy |
| `{SYMBOL}_probabilities.png` | Predicted probability of UP over the test period |
| `{SYMBOL}_comparison.png` | Actual vs. predicted direction over the test period |

---

## Project Structure

| File | Description |
|---|---|
| `main.py` | Entry point — CLI arguments and pipeline orchestration |
| `config.py` | Central configuration (dates, features, thresholds, paths) |
| `data_loader.py` | Downloads historical stock data via yfinance |
| `features.py` | Computes technical indicators and appends them to the dataframe |
| `dataset.py` | Builds feature matrix, target vector, and performs the time split |
| `model.py` | Trains the Random Forest and evaluates predictions |
| `reporting.py` | Prints metrics, feature importances, and sample predictions |
| `visualization.py` | Generates and saves charts to the results directory |
| `io_utilities.py` | Saves prediction CSVs and metrics text files |

---

## Technologies Used
- Python 3.13.3
- [yfinance](https://github.com/ranaroussi/yfinance)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)

---

## Disclaimer
This project is for educational purposes only and is not financial advice. Do not use the output of this model to make real investment decisions.

---

## Author
Edgard González