# ==============================
# Library Imports
# ==============================
import yfinance as yf


# ==============================
# Data Setup
# ==============================


def load_stock_data(symbol, start, end):

    """
    Load stock symbol data and return the corresponding dataframe, after dropping null values
    """
    
    df = yf.download(symbol, start=start, end=end)
    df = df.dropna()
    return df


