"""File saving utilities for StockChaser."""

# ==============================
# Library Imports
# ==============================
import os


# ==============================
# File Saving
# ==============================
def save_file(symbol, results, results_directory, baseline_accuracy, model_accuracy):
    """
    Save predictions and metrics for a given symbol to the results directory.
 
    Outputs:
        {symbol}_predictions.csv  — feature values with Actual and Predicted columns
        {symbol}_metrics.txt      — baseline and model accuracy scores
 
    Args:
        symbol            (str):          Stock ticker symbol.
        results           (pd.DataFrame): DataFrame with features, actuals, predictions.
        results_directory (str):          Path to the output directory.
        baseline_accuracy (float):        Naive baseline accuracy score.
        model_accuracy    (float):        Model accuracy score.
    """

    file_path = os.path.join(results_directory, f"{symbol}_predictions.csv")
    results.to_csv(file_path, index=True)

    print(f"Saved predictions to {file_path}")

    metrics_path = os.path.join(results_directory, f"{symbol}_metrics.txt")

    with open(metrics_path, "w") as f:
        f.write(f"Baseline Accuracy: {baseline_accuracy:.3f}\n")
        f.write(f"Model Accuracy: {model_accuracy:.3f}\n")

    print(f"Saved metrics to {metrics_path}")