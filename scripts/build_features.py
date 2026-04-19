#!/usr/bin/env python3
"""Multi-resolution feature engineering for crypto klines.

Per-symbol output (Pass 1):
  features/{resolution}/{symbol}.parquet   -- 23 columns per asset

Pass 2 (after all symbols): wide-join + BTC-vs-alt features -> final parquets.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

EWM_SPANS = (1, 5, 10, 30, 50)
BASIS_DEMEAN_DAYS = (1, 5, 10, 30, 50)
BASIS_DEVOL_DAYS = 20
ROLLING_WINDOW = 20
ZSCORE_WINDOW = 100
ZSCORE_MIN_PERIODS = 5
VOLUME_DIFF_EWM_SPAN = 20
EPS = 1e-12

OHLCV_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "taker_buy_base": "sum",
    "quote_volume": "sum",
    "trades": "sum",
}

RESOLUTIONS: dict[str, dict] = {
    "1m": {"freq": None, "bars_per_day": 1440},
    "1h": {"freq": "1h", "bars_per_day": 24},
    "8h": {"freq": "8h", "bars_per_day": 3},
}

FLOAT_COLS = (
    "open", "high", "low", "close",
    "volume", "taker_buy_base", "quote_volume",
)


def zscore_100bar(s: pd.Series) -> pd.Series:
    """Rolling 100-bar z-score: (x - mean_100) / std_100."""
    mu = s.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
    sigma = s.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std().clip(lower=EPS)
    return (s - mu) / sigma


def load_klines(root: Path, market: str, symbol: str) -> pd.DataFrame:
    """Load and concatenate all yearly parquet files for one market/symbol."""
    base = root / market / "klines" / "1m" / symbol
    files = sorted(base.glob("*.parquet"))
    if not files:
        print(f"  WARNING: no parquet files in {base}")
        return pd.DataFrame()
    frames = [pd.read_parquet(f, engine="pyarrow") for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"])

    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index.name = "timestamp"

    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    return df


def resample_bars(df_1m: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """Resample 1-min bars. Left-closed, right-labeled: [09:00,10:00) -> labeled 10:00."""
    if freq is None:
        return df_1m
    cols = list(OHLCV_AGG.keys())
    resampled = (
        df_1m[cols]
        .resample(freq, closed="left", label="right")
        .agg(OHLCV_AGG)
    )
    return resampled.dropna(subset=["close"])


def compute_base_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Compute 12 base features from futures perps OHLCV bars at one resolution.

    All features use only data at time <= t (no look-ahead).
    All are z-scored with 100-bar rolling window.
    """
    feat = pd.DataFrame(index=df.index)

    # --- Returns (5) ---
    # log_return at time t uses close[t] and close[t-1], both available at t
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for s in EWM_SPANS:
        # EWM at t uses log_ret[0..t], causal
        raw = log_ret.ewm(span=s, adjust=False).mean()
        feat[f"{symbol}_ewm_ret_{s}"] = zscore_100bar(raw)

    # --- Realized Volatility (1) ---
    # rolling_std at t uses log_ret[t-19..t], all available at t
    raw_rvol = log_ret.rolling(ROLLING_WINDOW).std()
    feat[f"{symbol}_realized_volatility"] = zscore_100bar(raw_rvol)

    # --- Normalized Volume (1) ---
    # rolling_mean at t uses volume[t-19..t], all available at t
    vol_mean = df["volume"].rolling(ROLLING_WINDOW).mean().clip(lower=EPS)
    raw_nvol = df["volume"] / vol_mean
    feat[f"{symbol}_norm_volume"] = zscore_100bar(raw_nvol)

    # --- Order Flow Imbalance (5) ---
    # imbalance at t uses taker_buy[t] and volume[t], both available at t
    imbalance = (2.0 * df["taker_buy_base"] - df["volume"]) / df["volume"].clip(lower=EPS)

    for s in EWM_SPANS:
        raw = imbalance.ewm(span=s, adjust=False).mean()
        feat[f"{symbol}_ewm_imbalance_{s}"] = zscore_100bar(raw)

    return feat


def compute_cross_market_features(
    fut_bars: pd.DataFrame,
    spot_bars: pd.DataFrame,
    symbol: str,
    bars_per_day: int,
) -> pd.DataFrame:
    """Compute 11 cross-market features from aligned futures and spot bars.

    All features use only data at time <= t (no look-ahead).
    """
    common = fut_bars.index.intersection(spot_bars.index)
    if common.empty:
        return pd.DataFrame()

    fut = fut_bars.loc[common]
    spot = spot_bars.loc[common]
    feat = pd.DataFrame(index=common)

    # --- Basis Z-scores (5) ---
    # basis at t uses fut_close[t] and spot_close[t], both available at t
    basis_raw = (fut["close"] - spot["close"]) / spot["close"].clip(lower=EPS)

    devol_win = max(BASIS_DEVOL_DAYS * bars_per_day, 2)
    basis_std = basis_raw.rolling(devol_win, min_periods=min(ZSCORE_MIN_PERIODS, devol_win)).std().clip(lower=EPS)

    for d in BASIS_DEMEAN_DAYS:
        demean_win = max(d * bars_per_day, 2)
        mp = min(ZSCORE_MIN_PERIODS, demean_win)
        basis_mean = basis_raw.rolling(demean_win, min_periods=mp).mean()
        feat[f"{symbol}_basis_zscore_{d}d"] = (basis_raw - basis_mean) / basis_std

    # --- Volume Diff (1) ---
    # volume_diff at t uses fut_volume[t] and spot_volume[t], available at t
    # EWM smoothing is causal (uses data up to t)
    volume_diff_raw = np.log(fut["volume"] + 1) - np.log(spot["volume"] + 1)
    volume_diff_smooth = volume_diff_raw.ewm(span=VOLUME_DIFF_EWM_SPAN, adjust=False).mean()
    feat[f"{symbol}_volume_diff_z"] = zscore_100bar(volume_diff_smooth)

    # --- Cross-Market Imbalance Diff (5) ---
    # imbalance at t uses buy[t] and volume[t] from each venue, available at t
    fut_imb = (2.0 * fut["taker_buy_base"] - fut["volume"]) / fut["volume"].clip(lower=EPS)
    spot_imb = (2.0 * spot["taker_buy_base"] - spot["volume"]) / spot["volume"].clip(lower=EPS)
    imb_diff = fut_imb - spot_imb

    for s in EWM_SPANS:
        raw = imb_diff.ewm(span=s, adjust=False).mean()
        feat[f"{symbol}_ewm_imbalance_diff_{s}"] = zscore_100bar(raw)

    return feat


def process_symbol(root: Path, symbol: str, out_dir: Path) -> None:
    """Run the full per-symbol pipeline for all 3 resolutions."""
    print(f"\n{'=' * 60}")
    print(f"Processing {symbol}")
    print(f"{'=' * 60}")

    print("Loading futures_um 1m bars...")
    fut_1m = load_klines(root, "futures_um", symbol)
    if fut_1m.empty:
        print(f"  SKIP {symbol}: no futures data")
        return
    print(f"  {len(fut_1m):,} rows  |  {fut_1m.index.min()} -> {fut_1m.index.max()}")

    print("Loading spot 1m bars...")
    spot_1m = load_klines(root, "spot", symbol)
    if spot_1m.empty:
        print(f"  WARNING: no spot data for {symbol}, cross-market features will be NaN")
    else:
        print(f"  {len(spot_1m):,} rows  |  {spot_1m.index.min()} -> {spot_1m.index.max()}")

    for res_name, res_cfg in RESOLUTIONS.items():
        freq = res_cfg["freq"]
        bpd = res_cfg["bars_per_day"]
        print(f"\n--- {res_name} (freq={freq or 'raw'}, {bpd} bars/day) ---")

        if freq is not None:
            print(f"  Resampling futures to {freq}...")
            fut_bars = resample_bars(fut_1m, freq)
            print(f"  Resampling spot to {freq}...")
            spot_bars = resample_bars(spot_1m, freq) if not spot_1m.empty else pd.DataFrame()
        else:
            fut_bars = fut_1m
            spot_bars = spot_1m

        print(f"  Futures bars: {len(fut_bars):,}")
        if not spot_bars.empty:
            print(f"  Spot bars:    {len(spot_bars):,}")

        print("  Computing 12 base features...")
        base = compute_base_features(fut_bars, symbol)

        print("  Computing 11 cross-market features...")
        if not spot_bars.empty:
            cross = compute_cross_market_features(fut_bars, spot_bars, symbol, bpd)
        else:
            cross_cols = (
                [f"{symbol}_basis_zscore_{d}d" for d in BASIS_DEMEAN_DAYS]
                + [f"{symbol}_volume_diff_z"]
                + [f"{symbol}_ewm_imbalance_diff_{s}" for s in EWM_SPANS]
            )
            cross = pd.DataFrame(index=fut_bars.index, columns=cross_cols, dtype="float64")

        features = base.join(cross, how="left")
        features = features.astype("float32")

        out_path = out_dir / res_name / f"{symbol}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out_path, engine="pyarrow")
        sz_mib = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {out_path}  ({sz_mib:.1f} MiB, {len(features):,} rows, {features.shape[1]} cols)")

        print(f"  Columns ({features.shape[1]}): {list(features.columns)}")
        nan_total = features.isna().sum()
        nan_any = nan_total[nan_total > 0]
        if nan_any.empty:
            print("  NaN: none")
        else:
            print(f"  NaN summary ({len(nan_any)}/{features.shape[1]} cols have NaN):")
            for c in nan_any.index:
                pct = 100 * nan_any[c] / len(features)
                print(f"    {c}: {nan_any[c]:,} ({pct:.2f}%)")

        valid = features.dropna()
        if len(valid) > 0:
            desc = valid.describe().T[["mean", "std", "min", "max"]]
            print("  Feature stats (valid rows only):")
            print(desc.to_string(float_format="{:.4f}".format))
        else:
            print("  WARNING: no valid (non-NaN) rows!")


def _worker(root: Path, symbol: str, out_dir: Path) -> str:
    """Wrapper for ProcessPoolExecutor."""
    process_symbol(root, symbol, out_dir)
    return symbol


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-resolution feature engineering")
    p.add_argument(
        "--root", type=Path,
        default=Path.home() / "Data" / "binance",
    )
    p.add_argument(
        "--symbols", default="BTCUSDT",
        help="comma-separated symbols (default: BTCUSDT)",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path.home() / "Data" / "binance" / "features",
    )
    p.add_argument(
        "--workers", type=int,
        default=min(os.cpu_count() or 1, 4),
        help="max parallel workers (default: min(cpu_count, 4))",
    )
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if len(symbols) == 1 or args.workers <= 1:
        for sym in symbols:
            process_symbol(args.root, sym, args.out)
    else:
        w = min(args.workers, len(symbols))
        print(f"Running {len(symbols)} symbols with {w} parallel workers")
        with ProcessPoolExecutor(max_workers=w) as pool:
            futures = {
                pool.submit(_worker, args.root, sym, args.out): sym
                for sym in symbols
            }
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    fut.result()
                    print(f"\n>>> {sym} completed.")
                except Exception as e:
                    print(f"\n>>> {sym} FAILED: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
