"""Model training and evaluation for StockChaser."""

# ==============================
# Library Imports
# ==============================
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ==============================
# Model Setup
# ==============================
def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on historical stock features.
    Hyperparameters are tuned via GridSearchCV with TimeSeriesSplit
    to respect the temporal ordering of the data and avoid leakage.

    No feature scaling is applied — Random Forests are scale-invariant
    by design, so normalization is not required.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series):    Training target vector.

    Returns:
        RandomForestClassifier: Best fitted model found by search.
    """

    param_grid = {
        "n_estimators":     [200, 300, 400],
        "max_depth":        [7, 10, 15],
        "min_samples_leaf": [10, 15, 20],
        "max_features":     ["sqrt", "log2"],
    }

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    # TimeSeriesSplit ensures folds respect chronological order,
    # preventing future data from leaking into earlier training folds.
    tscv = TimeSeriesSplit(n_splits=5)

    # zero_division=0 suppresses warnings when a fold predicts only one class
    scoring = make_scorer(f1_score, zero_division=0)

    search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1
    )

    search.fit(X_train, y_train.values.ravel())
    return search.best_estimator_


# ==============================
# Model Evaluation
# ==============================
def evaluate_model(model, X_test, y_test):
    """
    Generate predictions on the test set, compute accuracy,
    and produce a confusion matrix.

    Args:
        model  (RandomForestClassifier): Fitted model.
        X_test (pd.DataFrame):           Test feature matrix.
        y_test (pd.Series):              Test target vector.

    Returns:
        tuple: (predictions, accuracy, confusion_matrix)
    """

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return predictions, acc, matrix


# ==============================
# Baseline Accuracy
# ==============================
def baseline_accuracy(y_train, y_test):
    """
    Calculate the accuracy of a naive baseline that always predicts
    the majority class from the training set.

    Args:
        y_train (pd.Series): Training target vector.
        y_test  (pd.Series): Test target vector.

    Returns:
        float: Baseline accuracy score.
    """

    baseline_class = int((y_train.values.mean() > 0.5))
    predictions = np.full(len(y_test), baseline_class)
    acc = accuracy_score(y_test, predictions)
    return acc