"""Dataset preparation and time-series splitting for StockChaser."""

# ==============================
# Dataset Preparation
# ==============================
def prepare_dataset(df, feature_columns, predict_forward_days):
    """
    Build the feature matrix (X) and target vector (y).
 
    Target definition:
        1 -> closing price will be higher after predict_forward_days
        0 -> closing price will be lower after predict_forward_days
 
    The last N rows are removed since they cannot have a future label.
 
    Args:
        df                  (pd.DataFrame): DataFrame with features and Close price.
        feature_columns     (list[str]):    List of column names to use as features.
        predict_forward_days (int):         Number of days ahead to predict.
 
    Returns:
        tuple: (X, y) feature matrix and target vector.
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
    Split data chronologically without shuffling.
    Earlier data goes to training, later data goes to testing.
    This prevents data leakage across the time boundary.
 
    Args:
        X           (pd.DataFrame): Feature matrix.
        y           (pd.Series):    Target vector.
        train_ratio (float):        Proportion of data used for training (e.g. 0.8).
 
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test