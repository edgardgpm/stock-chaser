# ==============================
# Library Imports
# ==============================
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 

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


def print_feature_importances(model, feature_columns):
    """
    Print model feature importances if supported.
    """

    if not hasattr(model, "feature_importances_"):
        print("\nModel does not expose feature importances.")
        return

    print("\n--- Feature Importances ---")
    for name, importance in zip(feature_columns, model.feature_importances_):
        print(name, round(importance, 4))


def print_sample_predictions(X_test, y_test, predictions):
    """
    Display the first rows of predictions and return a dataframe combining features, actual labels, and model predictions.
    """

    results = X_test.copy()
    results["Actual"] = y_test
    results["Predicted"] = predictions

    print("\n--- Sample Predictions ---")
    print(results.head())

    return results


def print_confusion_matrix(matrix):
    """
    Print the confusion matrix returned during evaluation.
    """
    print("\n--- Confusion Matrix ---")
    print(matrix)
