"""Chart generation utilities for StockChaser."""

# ==============================
# Library Imports
# ==============================
import os
import matplotlib.pyplot as plt


# ==============================
# Constants
# ==============================
FIG_SIZE = (10, 5)


# ==============================
# Figure Plotting
# ==============================
def plot_figures(symbol, probabilities, actual, predictions, results_directory):
    """
    Generate and save two charts to the results directory:
        1. Predicted probability of UP over time.
        2. Actual vs. predicted direction comparison.
    """
     
    plt.figure(figsize=FIG_SIZE)
    plt.plot(probabilities)
    plt.title(f"{symbol} - Probability of UP")
    plt.xlabel("Time")
    plt.ylabel("Probability")

    plot_path = os.path.join(results_directory, f"{symbol}_probabilities.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved chart to {plot_path}")


    plt.figure(figsize=FIG_SIZE)
    plt.plot(actual.values, label="Actual")
    plt.plot(predictions, label="Predicted", alpha=0.7)
    plt.title(f"{symbol} - Actual vs. Predicted Direction")
    plt.xlabel("Time")
    plt.ylabel("Direction (0 = DOWN, 1 = UP)")
    plt.legend()

    plot_path = os.path.join(results_directory, f"{symbol}_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved chart to {plot_path}")