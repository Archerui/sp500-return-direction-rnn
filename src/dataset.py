import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_EXCLUDE_COLUMNS = {
    "date",
    "symbol",
    "future_5d_return",
    "target",
}


class StockSequenceDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split,
        lookback=30,
        train_end="2018-12-31",
        val_end="2021-12-31",
        feature_columns=None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.csv_path = csv_path
        self.split = split
        self.lookback = lookback
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

        if feature_columns is None:
            feature_columns = [
                col
                for col in df.columns
                if col not in DEFAULT_EXCLUDE_COLUMNS
            ]

        self.feature_columns = feature_columns

        print(f"[{split}] Number of feature columns:", len(self.feature_columns))
        print(f"[{split}] Feature columns:", self.feature_columns)

        self.X, self.y, self.dates, self.symbols = self._build_samples(df)

    def _get_split_mask(self, date):
        if self.split == "train":
            return date <= self.train_end
        elif self.split == "val":
            return (date > self.train_end) & (date <= self.val_end)
        else:
            return date > self.val_end

    def _build_samples(self, df):
        X_list = []
        y_list = []
        date_list = []
        symbol_list = []

        for symbol, group in df.groupby("symbol"):
            group = group.sort_values("date").reset_index(drop=True)

            features = group[self.feature_columns].values.astype(np.float32)
            targets = group["target"].values.astype(np.int64)
            dates = group["date"]

            for i in range(self.lookback, len(group)):
                label_date = dates.iloc[i]

                if not self._get_split_mask(label_date):
                    continue

                x = features[i - self.lookback : i]
                y = targets[i]

                X_list.append(x)
                y_list.append(y)
                date_list.append(label_date)
                symbol_list.append(symbol)

        if len(X_list) == 0:
            raise ValueError(
                f"No samples created for split={self.split}. "
                f"Check date split or input data."
            )

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list).astype(np.int64)

        return X, y, date_list, symbol_list

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def print_dataset_summary(dataset, name):
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("Number of samples:", len(dataset))
    print("X shape:", dataset.X.shape)
    print("y shape:", dataset.y.shape)

    unique, counts = np.unique(dataset.y, return_counts=True)
    print("Target distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(dataset.y) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")

    print("Date range:")
    print(min(dataset.dates), "to", max(dataset.dates))

    print("Number of symbols:")
    print(len(set(dataset.symbols)))

    print("Number of feature columns:")
    print(len(dataset.feature_columns))