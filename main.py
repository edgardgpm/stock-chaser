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
import config as cfg


# ==============================
# Full Pipeline Per Symbol
# ==============================
def run_pipeline_for_symbol(symbol, start_date, end_date, threshold):
    """
    Execute the full ML pipeline:
        load -> feature engineer -> build dataset ->
        split -> train -> evaluate
    """

    print(f"\n\n===== {symbol} =====\n")

    df = dl.load_stock_data(symbol, start_date, end_date)
    df = ft.add_features(df)

    X, y = ds.prepare_dataset(df, cfg.FEATURE_COLUMNS, cfg.PREDICT_FORWARD_DAYS)

    X_train, X_test, y_train, y_test = ds.time_series_split(X, y, cfg.TRAINING_SPLIT)

    model = md.train_model(X_train, y_train)

    baseline_acc = md.baseline_accuracy(y_train, y_test)
    model_pred, model_acc, matrix = md.evaluate_model(model, X_test, y_test)

    rp.evaluate_with_threshold(model, X_test, y_test, threshold)

    probabilities = model.predict_proba(X_test)[:, 1]

    rp.print_summary(baseline_acc, model_acc, y_train, y_test, model_pred)

    rp.print_feature_importances(model, cfg.FEATURE_COLUMNS)

    results = rp.print_sample_predictions(X_test, y_test, model_pred)

    rp.print_confusion_matrix(matrix)

    utils.save_file(symbol, results, cfg.RESULTS_DIR, baseline_acc, model_acc)

    vs.plot_figures(symbol,
                    probabilities,
                    y_test,
                    model_pred,
                    cfg.RESULTS_DIR)

    return {
        "baseline": baseline_acc,
        "accuracy": model_acc
    }


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":

    # Ensure RESULTS_DIR exists
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # Add arguments for use via CLI
    parser = argparse.ArgumentParser(description="Stock Direction Prediction")
    parser.add_argument("--symbol", type=str, default="AAPL",
                        help="Stock Ticker Symbol (e.g.: AAPL, MSFT...)")
    
    parser.add_argument("--start", type=str, default=cfg.START_DATE,
                        help="Start Date for Stock")
    
    parser.add_argument("--end", type=str, default=cfg.END_DATE,
                        help="End Date for Stock")

    parser.add_argument("--threshold", type=float, default=cfg.THRESHOLD,
                        help="Probability Threshold")

    args = parser.parse_args()

    results_arr = []


    # Run ML pipeline using the arguments provided
    res = run_pipeline_for_symbol(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        threshold=args.threshold
    )
