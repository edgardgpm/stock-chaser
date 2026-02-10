# ==============================
# Dataset Preparation
# ==============================
def prepare_dataset(df, feature_columns, predict_forward_days):
    """
    Build feature matrix (X) and target vector (y).

    Target:
        1 -> price will be higher after predict_forward_days
        0 -> price will be lower after predict_forward_days

    Remove the last rows that cannot have a future label.
    """

    X = df[feature_columns]
    y = (df["Close"].shift(-predict_forward_days) > df["Close"]).astype(int)

    X = X.iloc[:-predict_forward_days]
    y = y.iloc[:-predict_forward_days]

    return X, y


# ==============================
# Train/Test Split (Time Series Safe)
# ==============================
def time_series_split(X, y, train_ratio):
    """
    Split data without shuffling it.
    Ensure that earlier data -> training, later data -> testing.
    """

    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test