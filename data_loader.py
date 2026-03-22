"""Historical stock data retrieval for StockChaser."""

# ==============================
# Library Imports
# ==============================
import yfinance as yf


# ==============================
# Data Setup
# ==============================


def load_stock_data(symbol, start, end):
    """
    Download historical OHLCV data for a given ticker symbol via Yahoo Finance.
    Drops any rows with null values before returning.
 
    Args:
        symbol (str): Stock ticker symbol (e.g. 'AAPL').
        start  (str): Start date in YYYY-MM-DD format.
        end    (str): End date in YYYY-MM-DD format.
 
    Returns:
        pd.DataFrame: Clean historical price data.
 
    Raises:
        ValueError: If the ticker symbol is invalid or no data is returned.
        ConnectionError: If the download fails due to a network issue.
    """
    
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
    except Exception as e:
        raise ConnectionError(
            f"Failed to download data for '{symbol}'. "
            f"Check your internet connection. \nDetails: {e}"
        )

    if df.empty:
        raise ValueError(
            f"No data returned for '{symbol}' between {start} and {end}. "
            f"The ticker may be invalid or delisted."
        )
    
    df = df.dropna()

    if df.empty:
        raise ValueError(
            f"All rows were dropped after removing nulls for '{symbol}'. "
            f"Try a wider date range."
        )

    return df


