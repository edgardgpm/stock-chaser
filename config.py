"""Central configuration for StockChaser. All pipeline parameters are defined here."""

# ==============================
# Configuration
# ==============================

# Historical window used for training & evaluation
START_DATE = "2022-01-01"
END_DATE = "2026-06-01"

# Features selected after initial experimentation
FEATURE_COLUMNS = [
    "MA_5",
    "MA_10",
    "MA_Cross",
    "Return",
    "Return_2",
    "Return_5",
    "Volatility_5",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "Volume_Change",
]

# Predict whether price will be higher after N days
PREDICT_FORWARD_DAYS = 2

# Percentage of observations used for training (time-ordered)
TRAINING_SPLIT = 0.8

# Probability threshold for custom classification
# Set below 0.5 to predict UP more aggressively (higher recall, lower precision).
# Set above 0.5 to require more confidence before predicting UP (lower recall, higher precision).
THRESHOLD = 0.2
