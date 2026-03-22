"""Metrics reporting and console output for StockChaser."""

# ==============================
# Library Imports
# ==============================
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 


# ==============================
# Custom Threshold Evaluation
# ==============================
def evaluate_with_threshold(probabilities, y_test, threshold):
    """
    Evaluate predictions using a custom probability threshold instead
    of the default 0.5 decision boundary.
 
    Args:
        probabilities (np.ndarray): Predicted probabilities of UP (class 1).
        y_test        (pd.Series):  Test target vector.
        threshold     (float):      Probability cutoff for classifying as UP.
 
    Returns:
        str: Formatted evaluation report.
    """

    preds = (probabilities > threshold).astype(int)

    lines = [
        "\n--- Custom Threshold Evaluation ---",
        f"Accuracy: {accuracy_score(y_test, preds):.3f}",
        str(confusion_matrix(y_test, preds)),
        classification_report(y_test, preds)
    ]
 
    output = "\n".join(lines)
    print(output)
    return output


# ==============================
# Summary Metrics
# ==============================
def print_summary(baseline_acc, model_acc, y_train, y_test, model_pred):
    """
    Print dataset class balance and high-level performance numbers.
 
    Args:
        baseline_acc (float):      Naive baseline accuracy.
        model_acc    (float):      Model accuracy on the test set.
        y_train      (pd.Series):  Training target vector.
        y_test       (pd.Series):  Test target vector.
        model_pred   (np.ndarray): Model predictions on the test set.
 
    Returns:
        str: Formatted summary report.
    """

    lines = [
            "\n--- Class Balance ---",
            f"Training UP Ratio: {y_train.mean().item():.3f}",
            f"Testing UP Ratio: {y_test.mean().item():.3f}",
            f"Model Prediction UP Ratio: {model_pred.mean().item():.3f}",
            f"\nBaseline Accuracy: {baseline_acc:.3f}",
            f"Model Accuracy: {model_acc:.3f}",
    ]

    output = "\n".join(lines)
    print(output)
    return output


def print_feature_importances(model, feature_columns):
    """
    Print model feature importances if supported by the model.
 
    Args:
        model           (estimator):  Fitted model.
        feature_columns (list[str]):  List of feature names.
 
    Returns:
        str: Formatted feature importances report.
    """

    if not hasattr(model, "feature_importances_"):
        output = "\nModel does not expose feature importances."
        print(output)
        return output

    lines = ["\n--- Feature Importances ---"]
    for name, importance in zip(feature_columns, model.feature_importances_):
        lines.append(f"{name} {round(importance, 4)}")

    output = "\n".join(lines)
    print(output)
    return output


def print_sample_predictions(X_test, y_test, predictions):
    """
    Display the first rows of predictions and return a DataFrame
    combining features, actual labels, and model predictions.
 
    Args:
        X_test      (pd.DataFrame): Test feature matrix.
        y_test      (pd.Series):    Test target vector.
        predictions (np.ndarray):   Model predictions.
 
    Returns:
        tuple: (results DataFrame, formatted string output)
    """

    results = X_test.copy()
    results["Actual"] = y_test
    results["Predicted"] = predictions

    output = "\n--- Sample Predictions ---\n" + str(results.head())
    print(output)
    return results, output


def print_confusion_matrix(matrix):
    """
    Print the confusion matrix returned during model evaluation.
 
    Args:
        matrix (np.ndarray): Confusion matrix from sklearn.
 
    Returns:
        str: Formatted confusion matrix output.
    """

    output = "\n--- Confusion Matrix ---\n" + str(matrix)
    print(output)
    return output
