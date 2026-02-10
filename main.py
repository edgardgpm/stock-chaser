# ==============================
# Library Imports
# ==============================

# Standard data manipulation
import os
import argparse

# Project modules
import data_loader as dl
import features as ft
import model as md
import dataset as ds
import reporting as rp
import io_utilities as utils
import visualization as vs

# ==============================
# Save Output
# ==============================

RESULTS_DIR = "stockchaser_results"
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

    X, y = ds.prepare_dataset(df, FEATURE_COLUMNS, PREDICT_FORWARD_DAYS)

    X_train, X_test, y_train, y_test = ds.time_series_split(X, y, TRAINING_SPLIT)

    model = md.train_model(X_train, y_train)

    baseline_acc = md.baseline_accuracy(y_train, y_test)
    model_pred, model_acc, matrix = md.evaluate_model(model, X_test, y_test)

    rp.evaluate_with_threshold(model, X_test, y_test, threshold)

    probabilities = model.predict_proba(X_test)[:, 1]

    rp.print_summary(baseline_acc, model_acc, y_train, y_test, model_pred)

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

    utils.save_file(symbol, results, RESULTS_DIR, baseline_acc, model_acc)

    vs.plot_figures(symbol,
                    probabilities,
                    y_test,
                    model_pred,
                    RESULTS_DIR)

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
