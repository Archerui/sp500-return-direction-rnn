import os
import numpy as np
import pandas as pd
import yfinance as yf


RAW_STOCK_PATH = "data/raw/sp500_ohlcv.csv"
RAW_SPY_PATH = "data/raw/spy_ohlcv.csv"
OUTPUT_PATH = "data/processed/features_market.csv"

START_DATE = "2010-01-01"
END_DATE = None

HORIZON = 5
UP_THRESHOLD = 0.01
DOWN_THRESHOLD = -0.01

MIN_ROWS_PER_SYMBOL = 80


BASE_FEATURE_COLUMNS = [
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


MARKET_FEATURE_COLUMNS = [
    "spy_daily_return",
    "spy_momentum_5",
    "spy_momentum_10",
    "spy_volatility_10",
    "spy_volatility_20",
    "relative_return",
    "relative_momentum_5",
    "relative_momentum_10",
]


FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS


def download_spy_data():
    """
    Download SPY OHLCV data from Yahoo Finance.
    SPY is used as a market proxy for the S&P 500.
    """
    print("Downloading SPY data...")

    df = yf.download(
        "SPY",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise RuntimeError("SPY download failed. Empty dataframe returned.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    expected_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    df = df[expected_cols]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(RAW_SPY_PATH), exist_ok=True)
    df.to_csv(RAW_SPY_PATH, index=False)

    print("Saved SPY raw data to:", RAW_SPY_PATH)
    print("SPY shape:", df.shape)
    print("SPY date range:", df["date"].min(), "to", df["date"].max())

    return df


def load_or_download_spy_data():
    if os.path.exists(RAW_SPY_PATH):
        print("Loading existing SPY data from:", RAW_SPY_PATH)
        spy = pd.read_csv(RAW_SPY_PATH)
        spy["date"] = pd.to_datetime(spy["date"])
        return spy

    return download_spy_data()


def build_spy_features(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Build market-level SPY features by date.
    """
    spy = spy.sort_values("date").copy()

    close = spy["adj_close"]

    spy["spy_daily_return"] = close.pct_change()
    spy["spy_momentum_5"] = close / close.shift(5) - 1
    spy["spy_momentum_10"] = close / close.shift(10) - 1
    spy["spy_volatility_10"] = spy["spy_daily_return"].rolling(window=10).std()
    spy["spy_volatility_20"] = spy["spy_daily_return"].rolling(window=20).std()

    keep_cols = [
        "date",
        "spy_daily_return",
        "spy_momentum_5",
        "spy_momentum_10",
        "spy_volatility_10",
        "spy_volatility_20",
    ]

    spy_features = spy[keep_cols].copy()
    spy_features = spy_features.replace([np.inf, -np.inf], np.nan)
    spy_features = spy_features.dropna().reset_index(drop=True)

    print("SPY feature shape:", spy_features.shape)
    print("SPY feature date range:", spy_features["date"].min(), "to", spy_features["date"].max())

    return spy_features


def add_stock_features_for_symbol(group: pd.DataFrame) -> pd.DataFrame:
    """
    Create stock-level features and labels for one symbol.
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


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stock-vs-market relative features.
    """
    df = df.copy()

    df["relative_return"] = df["daily_return"] - df["spy_daily_return"]
    df["relative_momentum_5"] = df["momentum_5"] - df["spy_momentum_5"]
    df["relative_momentum_10"] = df["momentum_10"] - df["spy_momentum_10"]

    return df


def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["date", "symbol"] + FEATURE_COLUMNS + ["future_5d_return", "target"]

    df = df[required_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + ["future_5d_return", "target"])

    df["target"] = df["target"].astype(int)

    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("=" * 80)
    print("Loading stock data")
    print("=" * 80)

    df = pd.read_csv(RAW_STOCK_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    print("Raw stock shape:", df.shape)
    print("Stock date range:", df["date"].min(), "to", df["date"].max())
    print("Number of symbols:", df["symbol"].nunique())

    print("\nFiltering very short histories...")
    counts = df.groupby("symbol").size()
    valid_symbols = counts[counts >= MIN_ROWS_PER_SYMBOL].index
    df = df[df["symbol"].isin(valid_symbols)].copy()

    print("Remaining symbols:", df["symbol"].nunique())
    print("Shape after filtering:", df.shape)

    print("=" * 80)
    print("Loading / building SPY market features")
    print("=" * 80)

    spy = load_or_download_spy_data()
    spy_features = build_spy_features(spy)

    print("=" * 80)
    print("Generating stock features")
    print("=" * 80)

    featured_parts = []

    for symbol, group in df.groupby("symbol"):
        group = group.copy()
        group["symbol"] = symbol
        featured_parts.append(add_stock_features_for_symbol(group))

    featured = pd.concat(featured_parts, ignore_index=True)

    print("Stock feature shape before merge:", featured.shape)

    print("=" * 80)
    print("Merging stock features with SPY features")
    print("=" * 80)

    merged = featured.merge(spy_features, on="date", how="left")

    print("Merged shape:", merged.shape)
    print("Rows missing SPY features:", merged["spy_daily_return"].isna().sum())

    merged = add_relative_features(merged)

    print("=" * 80)
    print("Cleaning final feature data")
    print("=" * 80)

    processed = clean_feature_data(merged)

    processed = processed.sort_values(["symbol", "date"]).reset_index(drop=True)
    processed.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved market-enhanced features to:", OUTPUT_PATH)
    print("Processed shape:", processed.shape)
    print("Date range:", processed["date"].min(), "to", processed["date"].max())
    print("Number of symbols:", processed["symbol"].nunique())

    print("\nFeature count:", len(FEATURE_COLUMNS))
    print("Feature columns:")
    for col in FEATURE_COLUMNS:
        print("-", col)

    print("\nTarget distribution:")
    print(processed["target"].value_counts().sort_index())

    print("\nTarget distribution (%):")
    print((processed["target"].value_counts(normalize=True).sort_index() * 100).round(2))

    print("\nFirst 5 rows:")
    print(processed.head())


if __name__ == "__main__":
    main()