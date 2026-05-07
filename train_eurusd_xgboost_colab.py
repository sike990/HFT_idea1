"""
EUR/USD trend-start classifier with walk-forward backtesting (Colab friendly).

What this script does (simple words):
1) Downloads 5-minute EUR/USD OHLCV candles.
2) Builds technical features from candles.
3) Creates a target label: "trend start" vs "no trend".
4) Trains an XGBoost classifier month-by-month with walk-forward logic:
   - Train on months 1..12, test on month 13
   - Train on months 1..13, test on month 14
   - and so on.

You can run this directly in Google Colab.
"""

# ============================
# 1) Install and imports
# ============================
# If running in a fresh Colab runtime, uncomment:
# !pip install yfinance xgboost scikit-learn pandas numpy matplotlib

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================
# 2) Config you can edit easily
# ============================
@dataclass
class Config:
    symbol: str = "EURUSD=X"          # Yahoo Finance symbol for EUR/USD
    interval: str = "5m"              # 5-minute candles
    period: str = "60d"               # Yahoo 5m data is limited; use 60d chunks
    timezone: str = "UTC"

    # Label settings
    future_window: int = 8             # Minimum persistence threshold (8 candles)
    atr_mult: float = 1.2              # How strong the move must be vs ATR

    # Feature windows
    slope_window: int = 12
    atr_window: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    hhll_window: int = 20
    volume_window: int = 20

    # XGBoost params
    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.85
    colsample_bytree: float = 0.85


CFG = Config()


# ============================
# 3) Data download helper
# ============================
def download_recent_5m_history(symbol: str, timezone: str = "UTC", chunks: int = 12) -> pd.DataFrame:
    """
    Download recent 5m data in multiple 60-day chunks and combine them.

    NOTE:
    Yahoo Finance limits deep history for intraday 5m candles.
    This function stitches recent chunks; if Yahoo returns overlaps,
    duplicates are removed.
    """
    all_parts: List[pd.DataFrame] = []

    # Pull repeated recent chunks; deduplicate by index.
    # In practice this gives the maximum intraday coverage Yahoo allows.
    for _ in range(chunks):
        part = yf.download(
            tickers=symbol,
            period="60d",
            interval="5m",
            auto_adjust=False,
            progress=False,
            prepost=False,
        )
        if part is not None and not part.empty:
            all_parts.append(part)

    if not all_parts:
        raise ValueError("No data downloaded. Check internet or ticker symbol.")

    df = pd.concat(all_parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Keep standard OHLCV names
    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in downloaded data.")

    df = df[needed].copy()

    # Make timezone explicit
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(timezone)

    return df


# ============================
# 4) Feature engineering
# ============================
def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope of last `window` values using simple linear regression."""

    def _slope(x: np.ndarray) -> float:
        y = x
        x_idx = np.arange(len(y))
        if np.any(np.isnan(y)):
            return np.nan
        # polyfit degree 1 => slope
        return np.polyfit(x_idx, y, 1)[0]

    return series.rolling(window).apply(_slope, raw=True)


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    # Core helpers
    out["atr"] = compute_atr(out, cfg.atr_window)
    out["ret_1"] = out["Close"].pct_change(1)

    # 1) slope of recent closes
    out["slope_close"] = rolling_slope(out["Close"], cfg.slope_window)

    # 2) volume divergence: zscore(volume) - zscore(abs(return))
    vol_mean = out["Volume"].rolling(cfg.volume_window).mean()
    vol_std = out["Volume"].rolling(cfg.volume_window).std()
    ret_abs = out["ret_1"].abs()
    ret_mean = ret_abs.rolling(cfg.volume_window).mean()
    ret_std = ret_abs.rolling(cfg.volume_window).std()

    vol_z = (out["Volume"] - vol_mean) / vol_std
    ret_z = (ret_abs - ret_mean) / ret_std
    out["volume_divergence"] = vol_z - ret_z

    # 3) ATR-normalized range
    candle_range = out["High"] - out["Low"]
    out["atr_norm_range"] = candle_range / out["atr"]

    # 4) EMA separation
    out["ema_fast"] = out["Close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    out["ema_sep"] = (out["ema_fast"] - out["ema_slow"]) / out["atr"]

    # 5) higher-high / lower-low count in recent window
    hh = (out["High"] > out["High"].shift(1)).astype(float)
    ll = (out["Low"] < out["Low"].shift(1)).astype(float)
    out["higher_high_count"] = hh.rolling(cfg.hhll_window).sum()
    out["lower_low_count"] = ll.rolling(cfg.hhll_window).sum()

    # Basic lag features (often useful)
    for lag in [1, 2, 3, 6, 12]:
        out[f"ret_lag_{lag}"] = out["Close"].pct_change(lag)
        out[f"range_lag_{lag}"] = (out["High"].shift(lag) - out["Low"].shift(lag)) / out["atr"]

    return out


# ============================
# 5) Target label creation
# ============================
def create_trend_start_label(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Label = 1 if a new trend starts now and persists for at least 8 candles.

    Simple rule used:
    - Up-trend start if each of next N closes is above current close and
      the move after N candles is > atr_mult * ATR.
    - Down-trend start if each of next N closes is below current close and
      the move after N candles is > atr_mult * ATR (in absolute value).
    """
    n = cfg.future_window
    close = df["Close"]
    atr = df["atr"]

    up_flags = []
    down_flags = []

    for i in range(1, n + 1):
        up_flags.append(close.shift(-i) > close)
        down_flags.append(close.shift(-i) < close)

    up_persistent = pd.concat(up_flags, axis=1).all(axis=1)
    down_persistent = pd.concat(down_flags, axis=1).all(axis=1)

    up_move = (close.shift(-n) - close) > (cfg.atr_mult * atr)
    down_move = (close - close.shift(-n)) > (cfg.atr_mult * atr)

    label = ((up_persistent & up_move) | (down_persistent & down_move)).astype(int)
    return label


# ============================
# 6) Walk-forward monthly split
# ============================
def month_key(idx: pd.DatetimeIndex) -> pd.PeriodIndex:
    return idx.to_period("M")


def walk_forward_train_test(
    data: pd.DataFrame,
    feature_cols: List[str],
    min_train_months: int = 12,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window monthly walk-forward:
    train months [0..11] test month[12], then [0..12] test[13], etc.
    """
    periods = month_key(data.index)
    unique_months = periods.unique().sort_values()

    results = []
    all_preds = []

    for test_i in range(min_train_months, len(unique_months)):
        train_months = unique_months[:test_i]
        test_month = unique_months[test_i]

        train_mask = periods.isin(train_months)
        test_mask = periods == test_month

        train_df = data.loc[train_mask].copy()
        test_df = data.loc[test_mask].copy()

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test = test_df[feature_cols]
        y_test = test_df["target"]

        # Skip if train set has only one class
        if y_train.nunique() < 2:
            continue

        model = XGBClassifier(
            n_estimators=CFG.n_estimators,
            learning_rate=CFG.learning_rate,
            max_depth=CFG.max_depth,
            subsample=CFG.subsample,
            colsample_bytree=CFG.colsample_bytree,
            random_state=CFG.random_state,
            eval_metric="logloss",
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        fold_f1 = f1_score(y_test, pred, zero_division=0)
        fold_precision = precision_score(y_test, pred, zero_division=0)
        fold_recall = recall_score(y_test, pred, zero_division=0)

        results.append(
            {
                "test_month": str(test_month),
                "train_months": f"{train_months[0]} to {train_months[-1]}",
                "n_train": len(train_df),
                "n_test": len(test_df),
                "f1": fold_f1,
                "precision": fold_precision,
                "recall": fold_recall,
                "positive_rate_test": float(y_test.mean()),
            }
        )

        fold_preds = test_df[["target"]].copy()
        fold_preds["pred"] = pred
        fold_preds["test_month"] = str(test_month)
        all_preds.append(fold_preds)

    return pd.DataFrame(results), pd.concat(all_preds) if all_preds else pd.DataFrame()


# ============================
# 7) Main execution
# ============================
def main() -> None:
    print("Downloading EUR/USD 5-minute data...")
    raw = download_recent_5m_history(CFG.symbol, CFG.timezone)

    print(f"Raw candles: {len(raw)} | from {raw.index.min()} to {raw.index.max()}")

    feat = build_features(raw, CFG)
    feat["target"] = create_trend_start_label(feat, CFG)

    feature_cols = [
        "slope_close",
        "volume_divergence",
        "atr_norm_range",
        "ema_sep",
        "higher_high_count",
        "lower_low_count",
        "ret_1",
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_6",
        "ret_lag_12",
        "range_lag_1",
        "range_lag_2",
        "range_lag_3",
        "range_lag_6",
        "range_lag_12",
    ]

    # Drop rows with NaN after rolling and label shift
    data = feat.dropna(subset=feature_cols + ["target"]).copy()

    # Remove the tail where future label is not valid
    data = data.iloc[:-CFG.future_window] if len(data) > CFG.future_window else data

    print(f"Dataset after features/labels: {len(data)} rows")
    print(f"Positive class rate (trend start): {data['target'].mean():.4f}")

    results_df, preds_df = walk_forward_train_test(
        data=data,
        feature_cols=feature_cols,
        min_train_months=12,
    )

    if results_df.empty:
        print("No valid walk-forward folds. Need more monthly data.")
        return

    print("\nWalk-forward results by test month:")
    print(results_df)

    print("\nAverage metrics:")
    print(results_df[["f1", "precision", "recall"]].mean())

    if not preds_df.empty:
        print("\nOverall classification report:")
        print(classification_report(preds_df["target"], preds_df["pred"], zero_division=0))


if __name__ == "__main__":
    main()
