# ==============================
# Configuration
# ==============================

# Historical window used for training & evaluation
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# Features selected after initial experimentation
FEATURE_COLUMNS = [
    "MA_5",
    "MA_10",
    "Return",
    "Return_2",
    "Return_5",
    "Volatility_5"
]

# Predict whether price will be higher after N days
PREDICT_FORWARD_DAYS = 5

# Percentage of observations used for training (time-ordered)
TRAINING_SPLIT = 0.8

# Probability threshold for custom classification
THRESHOLD = 0.25

RESULTS_DIR = "stockchaser_results"
