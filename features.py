"""Feature engineering utilities for StockChaser."""

# ==============================
# Feature Setup
# ==============================
def add_features(df):
    """
    Compute technical indicators and append them as new columns
    to the existing DataFrame.

    Features added:
        - Return:        Daily percentage change in closing price
        - Return_2:      Return lagged by 2 days
        - Return_5:      Return lagged by 5 days
        - MA_5:          5-day simple moving average of closing price
        - MA_10:         10-day simple moving average of closing price
        - MA_Cross:      MA_5 minus MA_10 (crossover signal)
        - Volatility_5:  5-day rolling standard deviation of Return
        - RSI_14:        14-day Relative Strength Index (momentum oscillator)
        - MACD:          MACD line (EMA_12 minus EMA_26)
        - MACD_Signal:   9-day EMA of MACD line
        - MACD_Hist:     MACD histogram (MACD minus MACD_Signal)
        - Volume_Change: Daily percentage change in volume

    Rows with NaN values (introduced by rolling/shift operations)
    are dropped before returning.

    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame from data_loader.

    Returns:
        pd.DataFrame: Original DataFrame with feature columns appended.
    """

    df = df.copy()

    # --- Returns & Lags ---
    df["Return"] = df["Close"].pct_change()
    df["Return_2"] = df["Return"].shift(2)
    df["Return_5"] = df["Return"].shift(5)

    # --- Moving Averages & Crossover ---
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_Cross"] = df["MA_5"] - df["MA_10"]

    # --- Volatility ---
    df["Volatility_5"] = df["Return"].rolling(window=5).std()

    # --- RSI (14-day) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # --- Volume ---
    df["Volume_Change"] = df["Volume"].pct_change()

    df = df.dropna()
    return df