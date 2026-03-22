"""Feature engineering utilities for StockChaser."""

# ==============================
# Feature Setup
# ==============================
def add_features(df):
    """
    Compute technical indicators and append them as new columns
    to the existing DataFrame.
 
    Features added:
        - Return:       Daily percentage change in closing price
        - Return_2:     Return lagged by 2 days
        - Return_5:     Return lagged by 5 days
        - MA_5:         5-day simple moving average of closing price
        - MA_10:        10-day simple moving average of closing price
        - Volatility_5: 5-day rolling standard deviation of Return
 
    Rows with NaN values (introduced by rolling/shift operations)
    are dropped before returning.
 
    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame from data_loader.
 
    Returns:
        pd.DataFrame: Original DataFrame with feature columns appended.
    """

    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["Return_2"] = df["Return"].shift(2)
    df["Return_5"] = df["Return"].shift(5)

    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()

    df["Volatility_5"] = df["Return"].rolling(window=5).std()

    df = df.dropna()
    return df