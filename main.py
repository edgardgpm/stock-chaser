"""Entry point and pipeline orchestration for StockChaser."""

# ==============================
# Library Imports
# ==============================
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


# Base Directory Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# Full Pipeline Per Symbol
# ==============================
def run_pipeline_for_symbol(symbol, start_date, end_date, threshold, results_dir):
    """
    Execute the full ML pipeline:
        load -> feature engineer -> build dataset ->
        split -> train -> evaluate -> report -> save
        
    Args:
        symbol     (str):   Stock ticker symbol (e.g. 'AAPL').
        start_date (str):   Start date in YYYY-MM-DD format.
        end_date   (str):   End date in YYYY-MM-DD format.
        threshold  (float): Probability cutoff for classifying a prediction as UP.
 
    Returns:
        dict: Baseline accuracy and model accuracy for this symbol.
    """

    print(f"\n\n===== {symbol} =====\n")

    df = dl.load_stock_data(symbol, start_date, end_date)
    df = ft.add_features(df)

    X, y = ds.prepare_dataset(df, cfg.FEATURE_COLUMNS, cfg.PREDICT_FORWARD_DAYS)

    X_train, X_test, y_train, y_test = ds.time_series_split(X, y, cfg.TRAINING_SPLIT)

    model = md.train_model(X_train, y_train)

    baseline_acc = md.baseline_accuracy(y_train, y_test)
    model_pred, model_acc, matrix = md.evaluate_model(model, X_test, y_test)

    # Compute probabilities once and reuse across reporting and visualization
    probabilities = model.predict_proba(X_test)[:, 1]

    rp.evaluate_with_threshold(probabilities, y_test, threshold)

    rp.print_summary(baseline_acc, model_acc, y_train, y_test, model_pred)

    rp.print_feature_importances(model, cfg.FEATURE_COLUMNS)

    results, _ = rp.print_sample_predictions(X_test, y_test, model_pred)

    rp.print_confusion_matrix(matrix)

    utils.save_file(symbol, results, results_dir, baseline_acc, model_acc)

    vs.plot_figures(symbol,
                    probabilities,
                    y_test,
                    model_pred,
                    results_dir)

    return {
        "baseline": baseline_acc,
        "accuracy": model_acc
    }


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":

    # Ensure RESULTS_DIR exists
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Add arguments for use via CLI
    parser = argparse.ArgumentParser(description="Stock Direction Prediction")
    parser.add_argument("--symbol", type=str, default="AAPL",
                        help="Stock Ticker Symbol (e.g.: AAPL, MSFT...)")
    
    parser.add_argument("--start", type=str, default=cfg.START_DATE,
                        help="Start Date for Stock (YYYY-MM-DD)")
    
    parser.add_argument("--end", type=str, default=cfg.END_DATE,
                        help="End Date for Stock (YYYY-MM-DD)")

    parser.add_argument("--threshold", type=float, default=cfg.THRESHOLD,
                        help="Probability Threshold for UP Classification")

    args = parser.parse_args()

    # Run ML pipeline using the arguments provided
    try:
        res = run_pipeline_for_symbol(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            threshold=args.threshold,
            results_dir=results_dir
        )
    except (ValueError, ConnectionError) as e:
        print(f"\nError: {e}")
