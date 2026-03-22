"""Model training and evaluation for StockChaser."""

# ==============================
# Library Imports
# ==============================
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ==============================
# Model Setup
# ==============================
def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on historical stock features.
 
    No feature scaling is applied — Random Forests are scale-invariant
    by design, so normalization is not required.
 
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series):    Training target vector.
 
    Returns:
        RandomForestClassifier: Fitted model.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train.values.ravel())
    return model


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