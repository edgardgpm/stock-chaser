# ==============================
# Library Imports
# ==============================
import os
import matplotlib.pyplot as plt


# ==============================
# Figure Plotting
# ==============================
def plot_figures(symbol, probabilities, actual, predictions, results_directory):
    """
    Generate dataset figures for probability and stock comparison prior to saving on a results directory.
    """

    plt.figure(figsize=(10,5))
    plt.plot(probabilities)
    plt.title(f"{symbol} - Probability of UP")
    plt.xlabel("Time")
    plt.ylabel("Probability")

    plot_path = os.path.join(results_directory, f"{symbol}_probabilities.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved chart to {plot_path}")


    plt.figure(figsize=(10,5))
    plt.plot(actual.values, label="Actual")
    plt.plot(predictions, label="Predicted", alpha=0.7)
    plt.legend()

    plot_path = os.path.join(results_directory, f"{symbol}_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved chart to {plot_path}")