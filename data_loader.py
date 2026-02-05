# ==============================
# Library Imports
# ==============================
import yfinance as yf


# ==============================
# Data Setup
# ==============================

# Function to load stock symbol data and return the corresponding DataFrame (after dropping nulls)
def load_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df.dropna()
    return df


