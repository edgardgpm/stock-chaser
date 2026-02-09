# ==============================
# Feature Setup
# ==============================


def add_features(df):
    """
    Add new stock features to the existing dataframe.
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