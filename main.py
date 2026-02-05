# ==============================
# Library Imports
# ==============================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import data_loader as dl


# ==============================
# Symbol Setup
# ==============================
SYMBOL = "AAPL"
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"


dataFrame = dl.load_stock_data(SYMBOL, START_DATE, END_DATE)

print(dataFrame.head())