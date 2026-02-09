# ==============================
# Library Imports
# ==============================

# Standard data manipulation
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Project modules
import data_loader as dl
import features as ft
import model as md

# Evaluation helpers
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 


# ==============================
# Save Output
# ==============================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================
# Configuration
# ==============================

# Historical window used for training & evaluation
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# Features selected after initial experimentation
FEATURE_COLUMNS = [
    "MA_5",
    "MA_10",
    "Return",
    "Return_2",
    "Return_5",
    "Volatility_5"
]

# Predict whether price will be higher after N days
PREDICT_FORWARD_DAYS = 5

# Percentage of observations used for training (time-ordered)
TRAINING_SPLIT = 0.8

# Probability threshold for custom classification
THRESHOLD = 0.25


# ==============================
# Dataset Preparation
# ==============================
def prepare_dataset(df):
    """
    Build feature matrix (X) and target vector (y).

    Target:
        1 -> price will be higher after PREDICT_FORWARD_DAYS
        0 -> price will be lower after PREDICT_FORWARD_DAYS

    Remove the last rows that cannot have a future label.
    """

    X = df[FEATURE_COLUMNS]
    y = (df["Close"].shift(-PREDICT_FORWARD_DAYS) > df["Close"]).astype(int)

    X = X.iloc[:-PREDICT_FORWARD_DAYS]
    y = y.iloc[:-PREDICT_FORWARD_DAYS]

    return X, y


# ==============================
# Train/Test Split (Time Series Safe)
# ==============================
def time_series_split(X, y, train_ratio):
    """
    Split data without shuffling it.
    Ensure that earlier data -> training, later data -> testing.
    """

    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# ==============================
# Custom Threshold Evaluation
# ==============================
def evaluate_with_threshold(model, X_test, y_test, threshold):
    """
    Allow custom decision boundary for probability evaluation.
    """

    probabilities = model.predict_proba(X_test)[:, 1]
    preds = (probabilities > threshold).astype(int)

    print("\n--- Custom Threshold Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))


# ==============================
# Summary Metrics
# ==============================
def print_summary(baseline_acc, model_acc, y_train, y_test, model_pred):
    """
    Print dataset balance and high-level performance numbers.
    """

    print("\n--- Class Balance ---")
    print(f"Training UP Ratio: {y_train.mean().item():.3f}")
    print(f"Testing UP Ratio: {y_test.mean().item():.3f}")
    print(f"Model Prediction UP Ratio: {model_pred.mean().item():.3f}")

    print(f"\nBaseline Accuracy: {baseline_acc:.3f}")
    print(f"Model Accuracy: {model_acc:.3f}")


# ==============================
# Full Pipeline Per Symbol
# ==============================
def run_for_symbol(symbol, start_date, end_date, threshold):
    """
    Execute the full ML pipeline:
        load -> feature engineer -> build dataset ->
        split -> train -> evaluate
    """

    print(f"\n\n===== {symbol} =====\n")

    df = dl.load_stock_data(symbol, start_date, end_date)
    df = ft.add_features(df)

    X, y = prepare_dataset(df)

    X_train, X_test, y_train, y_test = time_series_split(X, y, TRAINING_SPLIT)

    model = md.train_model(X_train, y_train)

    baseline_acc = md.baseline_accuracy(y_train, y_test)
    model_pred, model_acc, matrix = md.evaluate_model(model, X_test, y_test)

    evaluate_with_threshold(model, X_test, y_test, threshold)

    print_summary(baseline_acc, model_acc, y_train, y_test, model_pred)

    print("\n--- Feature Importance ---")
    for name, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(name, round(importance, 4))


    results = X_test.copy()
    results["Actual"] = y_test
    results["Predicted"] = model_pred

    print("\nSample Predictions:")
    print(results.head())

    print("Confusion Matrix:")
    print(matrix)


    file_path = os.path.join(RESULTS_DIR, f"{symbol}_predictions.csv")
    results.to_csv(file_path, index=True)
    print(f"Saved predictions to {file_path}")

    metrics_path = os.path.join(RESULTS_DIR, f"{symbol}_metrics.txt")

    with open(metrics_path, "w") as f:
        f.write(f"Baseline Accuracy: {baseline_acc:.3f}\n")
        f.write(f"Model Acurracy: {model_acc:.3f}\n")

    print(f"Saved metrics to {metrics_path}")


    return {
        "baseline": baseline_acc,
        "accuracy": model_acc
    }


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":

    # Add arguments for use via CLI
    parser = argparse.ArgumentParser(description="Stock Direction Prediction")
    parser.add_argument("--symbol", type=str, default="AAPL",
                        help="Stock Ticker Symbol (e.g.: AAPL, MSFT...)")
    
    parser.add_argument("--start", type=str, default=START_DATE,
                        help="Start Date for Stock")
    
    parser.add_argument("--end", type=str, default=END_DATE,
                        help="End Date for Stock")

    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Probability Threshold")

    args = parser.parse_args()

    results_arr = []


    # Run ML pipeline using the arguments provided
    res = run_for_symbol(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        threshold=args.threshold
    )
