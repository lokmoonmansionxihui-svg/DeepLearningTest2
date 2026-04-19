#!/usr/bin/env python3
"""Download Binance Vision klines (monthly zips) → yearly Parquet.

Layout (default root ~/Data/binance):
  {root}/{market}/klines/{interval}/{symbol}/{year}.parquet

Markets per symbol:
  spot        — USDT spot (e.g. BTCUSDT)
  futures_um  — USDT-margined perpetual (e.g. BTCUSDT)

Missing months (404) are skipped; a year file is written only if ≥1 month exists.
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

import pandas as pd
from tqdm import tqdm

KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

MARKET_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("spot", "data/spot/monthly/klines"),
    ("futures_um", "data/futures/um/monthly/klines"),
)

DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "TRXUSDT",
    "DOGEUSDT",
    "ADAUSDT",
)


@dataclass(frozen=True)
class Market:
    key: str
    vision_klines_prefix: str
    symbol: str
    description: str


def build_markets(symbols: tuple[str, ...]) -> tuple[Market, ...]:
    markets: list[Market] = []
    for sym in symbols:
        for key, prefix in MARKET_TEMPLATES:
            markets.append(Market(key, prefix, sym, f"{key} {sym}"))
    return tuple(markets)


def monthly_zip_url(
    vision_prefix: str,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> str:
    mm = f"{month:02d}"
    return (
        "https://data.binance.vision/"
        f"{vision_prefix}/{symbol}/{interval}/{symbol}-{interval}-{year}-{mm}.zip"
    )


def fetch_month(
    vision_prefix: str,
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> pd.DataFrame | None:
    url = monthly_zip_url(vision_prefix, symbol, interval, year, month)
    req = urllib.request.Request(url, headers={"User-Agent": "CryptoDL/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    zf = zipfile.ZipFile(io.BytesIO(raw))
    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
    if not csv_names:
        raise ValueError(f"No CSV in zip: {url}")
    with zf.open(csv_names[0]) as f:
        raw = pd.read_csv(f, header=None, names=KLINES_COLUMNS, dtype="string")
    if len(raw) and raw.iloc[0, 0] == "open_time":
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.empty:
        return None
    for c in KLINES_COLUMNS:
        if c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        else:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").astype("int64")
    # Recent Binance Vision *spot* monthly CSVs use microseconds for open/close time; futures stay ms.
    if int(raw["open_time"].iloc[0]) >= 10**15:
        raw["open_time"] = (raw["open_time"] // 1000).astype("int64")
        raw["close_time"] = (raw["close_time"] // 1000).astype("int64")
    return raw


def calendar_months_in_year(year: int, last_year: int, last_month: int) -> range:
    if year < last_year:
        return range(1, 13)
    if year > last_year:
        return range(1, 1)
    return range(1, last_month + 1)


def download_market_year(
    market: Market,
    interval: str,
    year: int,
    months: range,
    out_path: Path,
    compression: str | None,
) -> bool:
    frames: list[pd.DataFrame] = []
    desc = f"{market.key}/{market.symbol} {year}"
    for m in tqdm(months, desc=desc, unit="mo", leave=False):
        df = fetch_month(market.vision_klines_prefix, market.symbol, interval, year, m)
        if df is not None:
            frames.append(df)
    if not frames:
        return False
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("open_time").drop_duplicates(subset=["open_time"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(
        out_path,
        index=False,
        engine="pyarrow",
        compression=compression,
    )
    return True


def default_end_ym() -> tuple[int, int]:
    now = datetime.now(tz=UTC)
    return now.year, now.month


def run_bundle(
    markets: tuple[Market, ...],
    interval: str,
    start_year: int,
    end_year: int,
    end_month: int,
    root: Path,
    compression: str | None,
    skip_existing: bool,
) -> None:
    for market in markets:
        base = root / market.key / "klines" / interval / market.symbol
        for year in range(start_year, end_year + 1):
            months = calendar_months_in_year(year, end_year, end_month)
            if months.start >= months.stop:
                continue
            out = base / f"{year}.parquet"
            if skip_existing and out.is_file():
                print(f"Skip existing {out}")
                continue
            ok = download_market_year(
                market, interval, year, months, out, compression
            )
            if ok:
                sz = out.stat().st_size / (1024 * 1024)
                print(f"Wrote {out} ({sz:.2f} MiB)")
            else:
                print(f"Skip {market.key} {market.symbol} {year} (no months)", file=sys.stderr)


def main() -> None:
    end_y, end_m = default_end_ym()
    p = argparse.ArgumentParser(description="Binance Vision klines → yearly Parquet")
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="comma-separated USDT symbols (default: all 8 coins)",
    )
    p.add_argument("--interval", default="1m")
    p.add_argument("--start-year", type=int, default=2018)
    p.add_argument("--end-year", type=int, default=end_y)
    p.add_argument("--end-month", type=int, default=end_m)
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "Data" / "binance",
        help="e.g. ~/Data/binance",
    )
    p.add_argument(
        "--parquet-compression",
        default="none",
        choices=("none", "snappy", "zstd", "gzip"),
        help="Parquet codec; default none = uncompressed",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="do not re-download years that already have a parquet file",
    )
    args = p.parse_args()
    compression: str | None
    if args.parquet_compression == "none":
        compression = None
    else:
        compression = args.parquet_compression

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    markets = build_markets(symbols)

    run_bundle(
        markets,
        args.interval,
        args.start_year,
        args.end_year,
        args.end_month,
        args.root,
        compression,
        args.skip_existing,
    )


if __name__ == "__main__":
    main()
