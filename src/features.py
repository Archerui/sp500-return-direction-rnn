import os
import numpy as np
import pandas as pd


RAW_PATH = "data/raw/sp500_ohlcv.csv"
OUTPUT_PATH = "data/processed/features.csv"

HORIZON = 5
UP_THRESHOLD = 0.01
DOWN_THRESHOLD = -0.01

MIN_ROWS_PER_SYMBOL = 80


FEATURE_COLUMNS = [
    "daily_return",
    "log_return",
    "open_close_return",
    "high_low_range",
    "volume_change",
    "ma_5_ratio",
    "ma_10_ratio",
    "ma_20_ratio",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "momentum_5",
    "momentum_10",
]


def add_features_for_symbol(group: pd.DataFrame) -> pd.DataFrame:
    """
    Create features and labels for one stock symbol.

    All rolling calculations are done within each symbol to avoid mixing data
    from different stocks.
    """
    group = group.sort_values("date").copy()

    close = group["adj_close"]
    open_ = group["open"]
    high = group["high"]
    low = group["low"]
    volume = group["volume"]

    group["daily_return"] = close.pct_change()
    group["log_return"] = np.log(close / close.shift(1))

    group["open_close_return"] = close / open_ - 1
    group["high_low_range"] = high / low - 1
    group["volume_change"] = volume.pct_change()

    ma_5 = close.rolling(window=5).mean()
    ma_10 = close.rolling(window=10).mean()
    ma_20 = close.rolling(window=20).mean()

    group["ma_5_ratio"] = close / ma_5 - 1
    group["ma_10_ratio"] = close / ma_10 - 1
    group["ma_20_ratio"] = close / ma_20 - 1

    group["volatility_5"] = group["daily_return"].rolling(window=5).std()
    group["volatility_10"] = group["daily_return"].rolling(window=10).std()
    group["volatility_20"] = group["daily_return"].rolling(window=20).std()

    group["momentum_5"] = close / close.shift(5) - 1
    group["momentum_10"] = close / close.shift(10) - 1

    # Future return from current day t to t + HORIZON.
    group["future_5d_return"] = close.shift(-HORIZON) / close - 1

    conditions = [
        group["future_5d_return"] < DOWN_THRESHOLD,
        (group["future_5d_return"] >= DOWN_THRESHOLD)
        & (group["future_5d_return"] <= UP_THRESHOLD),
        group["future_5d_return"] > UP_THRESHOLD,
    ]

    choices = [0, 1, 2]
    group["target"] = np.select(conditions, choices, default=np.nan)

    return group


def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean rows after feature generation.

    Rolling features, percentage changes, and future returns naturally create
    missing or infinite values near the beginning/end of each stock's history.
    """
    required_cols = ["date", "symbol"] + FEATURE_COLUMNS + ["future_5d_return", "target"]

    df = df[required_cols].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + ["future_5d_return", "target"])

    df["target"] = df["target"].astype(int)

    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    print("Raw shape:", df.shape)
    print("Number of symbols:", df["symbol"].nunique())

    print("\nFiltering very short histories...")
    counts = df.groupby("symbol").size()
    valid_symbols = counts[counts >= MIN_ROWS_PER_SYMBOL].index
    df = df[df["symbol"].isin(valid_symbols)].copy()

    print("Remaining symbols:", df["symbol"].nunique())
    print("Shape after filtering:", df.shape)

    print("\nGenerating features and labels...")
    featured_parts = []

    for symbol, group in df.groupby("symbol"):
        group = group.copy()
        group["symbol"] = symbol
        featured_parts.append(add_features_for_symbol(group))

    featured = pd.concat(featured_parts, ignore_index=True)

    print("Shape before cleaning:", featured.shape)

    processed = clean_feature_data(featured)

    processed = processed.sort_values(["symbol", "date"]).reset_index(drop=True)
    processed.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved processed features to:", OUTPUT_PATH)
    print("Processed shape:", processed.shape)
    print("Date range:", processed["date"].min(), "to", processed["date"].max())
    print("Number of symbols:", processed["symbol"].nunique())

    print("\nTarget distribution:")
    print(processed["target"].value_counts().sort_index())

    print("\nTarget distribution (%):")
    print((processed["target"].value_counts(normalize=True).sort_index() * 100).round(2))

    print("\nFeature columns:")
    for col in FEATURE_COLUMNS:
        print("-", col)

    print("\nFirst 5 rows:")
    print(processed.head())


if __name__ == "__main__":
    main()