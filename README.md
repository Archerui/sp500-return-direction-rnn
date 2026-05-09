# S&P 500 Return Direction Forecasting with Recurrent Neural Networks

A PyTorch-based project comparing traditional machine learning baselines and recurrent neural networks for multi-stock S&P 500 return direction classification.

The goal is not to predict exact stock prices. Instead, the project predicts whether a stock's future 5-day return will be:

- `down`
- `flat`
- `up`

This project uses a strict chronological split to avoid look-ahead bias.

---

## Project Overview

This project builds a full machine learning pipeline for stock return direction prediction:

1. Download historical S&P 500 stock OHLCV data.
2. Generate technical features from daily prices and volume.
3. Convert each stock into rolling 30-day sequences.
4. Train baseline models and recurrent neural networks.
5. Improve the GRU model with class-weighted loss.
6. Evaluate the final model on an out-of-sample test period.
7. Run a small feature ablation using SPY market features.

---

## Task Definition

For each S&P 500 stock, the model uses the previous 30 trading days of features to predict the stock's forward 5-day return direction.

### Input

Each sample is a sequence:

```text
30 trading days × 13 stock-level features
```

The stock-level features include returns, moving-average ratios, volatility, momentum, and volume change.

### Label

The target is based on future 5-day return:

```text
0 = down
1 = flat
2 = up
```

The final model is evaluated mainly with **macro-F1**, because the three classes are imbalanced and accuracy alone can be misleading.

---

## Data Split

The main experiments use the following chronological split:

| Split | Period |
|---|---|
| Train | 2010-2018 |
| Validation | 2019-2021 |
| Test | 2022-2026 |

The validation set is used for model selection. The test set is only used for the final out-of-sample evaluation.

---

## Repository Structure

```text
sp500-return-direction-rnn/
│  README.md
│  requirements.txt
│  LICENSE
│
├─configs
│      default.yaml
│
├─notebooks
│      01_baseline_models.ipynb
│      02_train_rnn.ipynb
│      03_train_lstm.ipynb
│      04_train_weighted_gru_final.ipynb
│      05_ablation_market_features.ipynb
│      06_final_results_summary.ipynb
│
├─outputs
│  ├─figures
│  └─metrics
│
└─src
       dataset.py
       features.py
       features_market.py
       models.py
       utils.py
```

Large data files and checkpoints are excluded from GitHub.

---

## Setup

Create and activate a conda environment:

```bash
conda create -n stock python=3.11
conda activate stock
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build that matches your system.

---

## Data Preparation

The raw and processed data files are not included in this repository.

Expected generated files:

```text
data/raw/sp500_ohlcv.csv
data/raw/spy_ohlcv.csv
data/processed/features.csv
data/processed/features_market.csv
```

To reproduce the data locally:

```bash
python download_sp500.py
python src/features.py
python src/features_market.py
```

The main model uses:

```text
data/processed/features.csv
```

The market-feature ablation uses:

```text
data/processed/features_market.csv
```

---

## Notebooks

Run the notebooks in this order:

| Notebook | Purpose |
|---|---|
| `01_baseline_models.ipynb` | Majority class, logistic regression, random forest |
| `02_train_rnn.ipynb` | Vanilla RNN baseline |
| `03_train_lstm.ipynb` | LSTM baseline |
| `04_train_weighted_gru_final.ipynb` | Final class-weighted GRU model |
| `05_ablation_market_features.ipynb` | SPY market feature ablation |
| `06_final_results_summary.ipynb` | Final result tables and figures |

---

## Models

The project compares:

- Majority class baseline
- Logistic regression
- Random forest
- Vanilla RNN
- LSTM
- Weighted GRU

The final model is a single-layer GRU trained with class-weighted cross entropy.

Final GRU configuration:

| Parameter | Value |
|---|---:|
| Lookback window | 30 |
| Input dimension | 13 |
| Hidden dimension | 128 |
| GRU layers | 1 |
| Dropout | 0.3 |
| Batch size | 1024 |
| Learning rate | 0.001 |
| Epochs | 8 |

---

## Results

### Validation Model Comparison

| model               | split      |   accuracy |   macro_f1 |   weighted_f1 |     down_f1 |    flat_f1 |      up_f1 |
|:--------------------|:-----------|-----------:|-----------:|--------------:|------------:|-----------:|-----------:|
| Weighted GRU        | validation |   0.403672 |   0.39351  |      0.405835 |   0.369676  |   0.349338 |   0.461515 |
| Logistic Regression | validation |   0.434333 |   0.32264  |      0.367895 | nan         | nan        | nan        |
| LSTM                | validation |   0.439169 |   0.315079 |      0.357146 |   0.122656  |   0.233312 |   0.589268 |
| RNN                 | validation |   0.445403 |   0.295099 |      0.342    |   0.0773887 |   0.205907 |   0.602001 |
| Random Forest       | validation |   0.444958 |   0.293883 |      0.344021 | nan         | nan        | nan        |
| Majority Class      | validation |   0.446325 |   0.205728 |      0.275465 | nan         | nan        | nan        |

The weighted GRU has the best validation macro-F1. Some models have higher accuracy, but they are more biased toward the majority class. Macro-F1 is more useful here because each class should matter equally.

### Final Weighted GRU: Validation vs Test

| split      |   accuracy |   macro_f1 |   weighted_f1 |   down_f1 |   flat_f1 |    up_f1 |
|:-----------|-----------:|-----------:|--------------:|----------:|----------:|---------:|
| validation |   0.403672 |   0.39351  |      0.405835 |  0.369676 |  0.349338 | 0.461515 |
| test       |   0.389713 |   0.368519 |      0.38586  |  0.367929 |  0.278346 | 0.459282 |

The test macro-F1 is lower than validation macro-F1, which is expected in a strict time-based split. The flat class is the hardest class to predict.

---

## Market Feature Ablation

I also tested adding SPY market features and relative return features.

Added features include:

- SPY daily return
- SPY momentum
- SPY volatility
- Stock return relative to SPY
- Stock momentum relative to SPY

This did not improve test macro-F1 in my experiment. The final model therefore uses only stock-level features.

---

## Key Takeaways

- Class-weighted loss improved the GRU's macro-F1 compared with the unweighted sequence models.
- Accuracy alone was not a good metric because the model could get high accuracy by predicting the majority class.
- The flat class was consistently the hardest class.
- Adding simple SPY market features did not improve the final test result.
- The project shows a realistic time-series ML workflow rather than a highly optimized trading strategy.

---

## Reproducibility Notes

The data files and model checkpoints are not committed to GitHub.

Recommended `.gitignore` entries:

```gitignore
data/raw/*.csv
data/processed/*.csv
outputs/checkpoints/
outputs/predictions/
*.pt
*.pth
*.joblib
*.pkl
```

Metrics and figures can be committed because they are small and useful for reviewing results.

---

## Disclaimer

This project is for educational purposes only. It is not financial advice and should not be used for real trading decisions.
