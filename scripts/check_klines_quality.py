#!/usr/bin/env python3
"""Scan Binance kline Parquet files for gaps, bad values, and extreme 1m moves."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_MS = 60_000  # 1m bars


def check_file(path: Path, jump_logret_thresh: float) -> dict:
    df = pd.read_parquet(path, engine="pyarrow")
    n = len(df)
    out: dict = {"path": str(path), "rows": n}

    if n == 0:
        out["error"] = "empty"
        return out

    # Nulls
    nulls = df.isna().sum()
    out["null_cols"] = {c: int(nulls[c]) for c in nulls.index if nulls[c] > 0}

    # Sort & dupes on open_time
    dupes = int(df["open_time"].duplicated().sum())
    out["dup_open_time"] = dupes

    ts = df["open_time"].to_numpy(dtype=np.int64)
    if not (np.diff(ts) > 0).all():
        out["open_time_not_strictly_increasing"] = True

    # Monotonic gaps (missing minutes)
    if n > 1:
        deltas = np.diff(ts)
        if len(np.unique(deltas)) == 1 and int(deltas[0]) == 60_000_000:
            out["time_unit_issue"] = (
                "all deltas 60_000_000 µs — spot Vision zips use µs; "
                "divide open_time/close_time by 1000 (fixed in download script)"
            )
        expected = deltas == EXPECTED_MS
        out["unexpected_delta_count"] = int((~expected).sum())
        bad = deltas[~expected]
        if len(bad) > 0:
            out["delta_lt_1m_count"] = int((bad < EXPECTED_MS).sum())
            out["delta_gt_1m_count"] = int((bad > EXPECTED_MS).sum())
            # integer-minute gaps → missing bars = mult - 1
            mult = (bad // EXPECTED_MS).astype(np.int64)
            rem = bad % EXPECTED_MS
            irregular = rem != 0
            out["delta_not_multiple_of_1m_count"] = int(irregular.sum())
            whole = mult[~irregular]
            if len(whole):
                out["gap_multiples_max"] = int(whole.max())
                out["total_missing_minutes_est"] = int(np.sum(np.maximum(0, whole - 1)))
            else:
                out["total_missing_minutes_est"] = 0
        else:
            out["total_missing_minutes_est"] = 0
    else:
        out["unexpected_delta_count"] = 0
        out["total_missing_minutes_est"] = 0

    # OHLC consistency
    bad_high = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
    bad_low = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
    out["ohlc_high_inconsistent"] = bad_high
    out["ohlc_low_inconsistent"] = bad_low

    # Signs / zeros
    for col in ("open", "high", "low", "close"):
        out[f"{col}_le_zero"] = int((df[col] <= 0).sum())
    out["volume_negative"] = int((df["volume"] < 0).sum())
    out["volume_zero"] = int((df["volume"] == 0).sum())
    out["volume_zero_pct"] = round(100.0 * out["volume_zero"] / n, 4)
    out["quote_volume_negative"] = int((df["quote_volume"] < 0).sum())
    out["trades_negative"] = int((df["trades"] < 0).sum())
    out["trades_zero"] = int((df["trades"] == 0).sum())

    # 1m log-return on close (extreme jumps)
    c = df["close"].to_numpy(dtype=np.float64)
    prev = c[:-1]
    cur = c[1:]
    valid = prev > 0
    lr = np.full(len(c) - 1, np.nan)
    lr[valid] = np.log(cur[valid] / prev[valid])
    abs_lr = np.abs(lr)
    mask = np.isfinite(abs_lr)
    if mask.any():
        mx = float(np.nanmax(abs_lr))
        out["max_abs_logret_1m"] = mx
        jumps = int((abs_lr > jump_logret_thresh).sum())
        out[f"jumps_logret_gt_{jump_logret_thresh}"] = jumps
        if jumps > 0 and jumps <= 20:
            idx = np.where(abs_lr > jump_logret_thresh)[0] + 1
            samples = []
            for i in idx[:10]:
                samples.append(
                    {
                        "open_time": int(df["open_time"].iloc[i]),
                        "close_prev": float(df["close"].iloc[i - 1]),
                        "close": float(df["close"].iloc[i]),
                        "logret": float(lr[i - 1]),
                    }
                )
            out["jump_samples"] = samples
    else:
        out["max_abs_logret_1m"] = None

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "Data" / "binance",
    )
    p.add_argument(
        "--jump-logret",
        type=float,
        default=0.15,
        help="flag 1m |log(close/close_prev)| above this (e.g. 0.15 ~ 16%% move)",
    )
    args = p.parse_args()
    files = sorted(args.root.rglob("*.parquet"))
    if not files:
        print(f"No parquet under {args.root}")
        return
    print(f"Files: {len(files)}  root={args.root}\n")
    for f in files:
        r = check_file(f, args.jump_logret)
        rel = f.relative_to(args.root)
        print(f"=== {rel}  rows={r.get('rows')}")
        if "error" in r:
            print(f"  {r['error']}")
            continue
        skip = {"path", "rows", "jump_samples"}
        for k, v in sorted(r.items()):
            if k in skip:
                continue
            if v == 0 or v == {} or v is False:
                continue
            print(f"  {k}: {v}")
        if r.get("jump_samples"):
            print("  jump_samples (up to 10):")
            for s in r["jump_samples"]:
                print(f"    {s}")
        print()


if __name__ == "__main__":
    main()
