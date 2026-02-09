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
    """

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Use the model to make predictions on the test values, calculate accuracy of the model, and generate the confusion matrix for the model.
    """

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return predictions, acc, matrix


def baseline_accuracy(y_train, y_test):
    """
    Calculate accuracy of the baseline.
    """
    
    baseline_class = int(y_train.mean() > 0.5)
    predictions = np.full(len(y_test), baseline_class)
    acc = accuracy_score(y_test, predictions)
    return acc