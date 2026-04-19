#!/usr/bin/env python3
"""Single-phase training: Neg-Sharpe or Mean–Variance loss on 15-min rebalance.

Every 15 minutes:
  - Multi-resolution features → position = tanh(signal) ∈ [-1, 1]
  - Gross PnL = position × 15min_return (bps)
  - Cost = cost_bps × |Δposition| at rebalance boundaries only
  - Net PnL = gross − cost

Loss modes (see --loss):
  - sharpe:  minimize −mean(net_pnl) / (std(net_pnl) + ε)   [scale-free, no λ]
  - mv:      minimize −(mean(net_pnl) − λ·var(net_pnl))      [λ = risk aversion]

Usage:
    uv run python scripts/train.py --symbol BTCUSDT --loss sharpe
    uv run python scripts/train.py --symbol BTCUSDT --loss mv --lambda-risk 2.0
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Constants ──────────────────────────────────────────────────────────────────

F = 23
LOOKBACK_8H = 21
LOOKBACK_1H = 72
LOOKBACK_1M = 120
REBAL_INTERVAL = 15  # rebalance every 15 minutes
PERIODS_PER_YEAR = 365.25 * 24 * 60 / REBAL_INTERVAL  # 35040

SPLITS = {
    "train": ("2020-01-01", "2023-05-31"),
    "val": ("2023-06-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-05-31"),
    "paper": ("2025-06-01", "2026-12-31"),
}

WARMUP_DAYS = 7


# ── Data loading ──────────────────────────────────────────────────────────────


@dataclass
class TimeSeriesStore:
    feat_1m: np.ndarray   # (N, F) float32
    feat_1h: np.ndarray   # (H, F) float32
    feat_8h: np.ndarray   # (E, F) float32
    ts_1m: np.ndarray     # (N,) int64  epoch seconds
    ts_1h: np.ndarray
    ts_8h: np.ndarray
    close_1m: np.ndarray  # (N,) float64
    h_idx: np.ndarray     # (N,) int64
    e_idx: np.ndarray     # (N,) int64
    label_mean: float = 0.0
    label_std: float = 1.0


def load_store(feature_dir: Path, kline_root: Path, symbol: str) -> TimeSeriesStore:
    print("Loading features and close prices...")
    f1m = pd.read_parquet(feature_dir / "1m" / f"{symbol}.parquet")
    f1h = pd.read_parquet(feature_dir / "1h" / f"{symbol}.parquet")
    f8h = pd.read_parquet(feature_dir / "8h" / f"{symbol}.parquet")

    base = kline_root / "futures_um" / "klines" / "1m" / symbol
    files = sorted(base.glob("*.parquet"))
    raw = pd.concat([pd.read_parquet(f, engine="pyarrow") for f in files], ignore_index=True)
    raw = raw.sort_values("open_time").drop_duplicates(subset=["open_time"])
    raw.index = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    close_series = pd.to_numeric(raw["close"], errors="coerce").astype("float64")
    close_aligned = close_series.reindex(f1m.index).ffill()

    ts_1m = f1m.index.astype("int64").values // 1000
    ts_1h = f1h.index.astype("int64").values // 1000
    ts_8h = f8h.index.astype("int64").values // 1000
    h_idx = np.searchsorted(ts_1h, ts_1m, side="right") - 1
    e_idx = np.searchsorted(ts_8h, ts_1m, side="right") - 1

    print(f"  1m: {f1m.shape}, 1h: {f1h.shape}, 8h: {f8h.shape}")
    return TimeSeriesStore(
        feat_1m=f1m.values.astype(np.float32),
        feat_1h=f1h.values.astype(np.float32),
        feat_8h=f8h.values.astype(np.float32),
        ts_1m=ts_1m, ts_1h=ts_1h, ts_8h=ts_8h,
        close_1m=close_aligned.values,
        h_idx=h_idx, e_idx=e_idx,
    )


def compute_rebal_indices(store: TimeSeriesStore, split: str) -> np.ndarray:
    """Valid 1m indices at 15-min rebalance points within a split."""
    start_s = int(pd.Timestamp(SPLITS[split][0], tz="UTC").timestamp())
    end_s = int(pd.Timestamp(SPLITS[split][1] + " 23:59:59", tz="UTC").timestamp())

    N = len(store.ts_1m)
    mask = np.ones(N, dtype=bool)
    mask &= store.ts_1m >= start_s
    mask &= store.ts_1m <= end_s
    mask &= np.arange(N) >= (LOOKBACK_1M - 1)
    mask &= store.h_idx >= (LOOKBACK_1H - 1)
    mask &= store.e_idx >= (LOOKBACK_8H - 1)
    mask &= np.arange(N) + REBAL_INTERVAL < N
    warmup_s = store.ts_1m[0] + WARMUP_DAYS * 86400
    mask &= store.ts_1m >= warmup_s

    all_valid = np.where(mask)[0]
    # subsample to every REBAL_INTERVAL minutes
    indices = all_valid[::REBAL_INTERVAL]

    if len(indices) == 0:
        print(f"  Split '{split}': 0 valid rebalance points!")
        return indices

    t0 = pd.Timestamp(store.ts_1m[indices[0]], unit="s", tz="UTC")
    t1 = pd.Timestamp(store.ts_1m[indices[-1]], unit="s", tz="UTC")
    print(f"  Split '{split}': {len(indices):,} rebalance points ({t0} -> {t1})")
    return indices


# ── Dataset ───────────────────────────────────────────────────────────────────


class CryptoDataset(Dataset):
    """Each sample = one 15-min rebalance point.

    Returns (x_8h, x_1h, x_1m, fwd_ret_15m_bps, has_nan)
    """

    def __init__(self, store: TimeSeriesStore, indices: np.ndarray):
        self.s = store
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]

        x_1m = self.s.feat_1m[i - LOOKBACK_1M + 1: i + 1]
        h = self.s.h_idx[i]
        x_1h = self.s.feat_1h[h - LOOKBACK_1H + 1: h + 1]
        e = self.s.e_idx[i]
        x_8h = self.s.feat_8h[e - LOOKBACK_8H + 1: e + 1]

        c_now = self.s.close_1m[i]
        c_fwd = self.s.close_1m[i + REBAL_INTERVAL]
        fwd_ret = np.log(c_fwd / c_now) * 10000 if c_now > 0 else 0.0  # bps

        has_nan = (
            np.isnan(x_1m).any() or np.isnan(x_1h).any() or np.isnan(x_8h).any()
        )

        return (
            torch.from_numpy(x_8h),
            torch.from_numpy(x_1h),
            torch.from_numpy(x_1m),
            torch.tensor(fwd_ret, dtype=torch.float32),
            torch.tensor(has_nan, dtype=torch.bool),
        )


def collate_fn(batch):
    x8h, x1h, x1m, ret, has_nan = zip(*batch)
    return (
        torch.stack(x8h),
        torch.stack(x1h),
        torch.stack(x1m),
        torch.stack(ret),
        torch.stack(has_nan),
    )


# ── Model ─────────────────────────────────────────────────────────────────────


class FeatureDropout(nn.Module):
    def __init__(self, p: float = 0.15):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(
            torch.full((1, 1, x.shape[2]), 1 - self.p, device=x.device)
        )
        return x * mask / (1 - self.p)


class MultiBranchLSTM(nn.Module):
    def __init__(self, f: int = F, hidden: int = 32, feat_drop_p: float = 0.15):
        super().__init__()
        self.feat_drop = FeatureDropout(p=feat_drop_p)

        self.lstm_8h = nn.LSTM(f, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm_1h = nn.LSTM(f, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.lstm_1m = nn.LSTM(f, hidden, num_layers=1, batch_first=True)

        concat_dim = hidden * 3
        self.head = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

        # initialize last linear so initial |position| ≈ 0.3-0.5 (tanh linear region)
        nn.init.xavier_normal_(self.head[-1].weight, gain=0.5)

    def forward(self, x_8h, x_1h, x_1m) -> torch.Tensor:
        """Returns position in [-1, 1]."""
        x_8h = torch.nan_to_num(self.feat_drop(x_8h), 0.0)
        x_1h = torch.nan_to_num(self.feat_drop(x_1h), 0.0)
        x_1m = torch.nan_to_num(self.feat_drop(x_1m), 0.0)

        _, (h_8h, _) = self.lstm_8h(x_8h)
        _, (h_1h, _) = self.lstm_1h(x_1h)
        _, (h_1m, _) = self.lstm_1m(x_1m)

        combined = torch.cat([h_8h[-1], h_1h[-1], h_1m[-1]], dim=1)
        signal = self.head(combined).squeeze(-1)
        return torch.tanh(signal)


# ── Loss ──────────────────────────────────────────────────────────────────────


def _net_pnl_from_positions(
    positions: torch.Tensor,
    fwd_ret_bps: torch.Tensor,
    prev_position: torch.Tensor,
    cost_coeff: float,
) -> torch.Tensor:
    """Shared: net PnL per step (same as evaluate)."""
    gross_pnl = positions * fwd_ret_bps
    all_pos = torch.cat([prev_position, positions])
    turnover = torch.abs(all_pos[1:] - all_pos[:-1])
    return gross_pnl - cost_coeff * turnover


class NegSharpeLoss(nn.Module):
    """Negative sample Sharpe on net_pnl within each chunk (unbiased std, ddof=1).

    loss = -mean(net_pnl) / (std(net_pnl) + eps)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        positions: torch.Tensor,
        fwd_ret_bps: torch.Tensor,
        prev_position: torch.Tensor,
        cost_coeff: float,
    ) -> torch.Tensor:
        net_pnl = _net_pnl_from_positions(
            positions, fwd_ret_bps, prev_position, cost_coeff
        )
        # ddof=1 matches typical sample Sharpe; need T>=2 for finite std
        if net_pnl.numel() < 2:
            return -net_pnl.mean() / (self.eps + 1e-8)
        std = net_pnl.std(unbiased=True)
        return -net_pnl.mean() / (std + self.eps)


class MeanVarianceLoss(nn.Module):
    """Mean–variance utility on net_pnl (Markowitz-style scalar objective).

    loss = -(E[net_pnl] − λ·Var[net_pnl])

    λ (lambda_risk): larger → more penalty on variance → smoother / smaller positions
    often; **not** scale-invariant: scaling all positions by c scales E by c and Var by c².

    Unbiased variance (ddof=1) when T≥2.
    """

    def __init__(self, lambda_risk: float = 2.0):
        super().__init__()
        self.lambda_risk = lambda_risk

    def forward(
        self,
        positions: torch.Tensor,
        fwd_ret_bps: torch.Tensor,
        prev_position: torch.Tensor,
        cost_coeff: float,
    ) -> torch.Tensor:
        net_pnl = _net_pnl_from_positions(
            positions, fwd_ret_bps, prev_position, cost_coeff
        )
        if net_pnl.numel() < 2:
            var = torch.zeros((), device=net_pnl.device, dtype=net_pnl.dtype)
        else:
            var = net_pnl.var(unbiased=True)
        return -(net_pnl.mean() - self.lambda_risk * var)


# ── Training ──────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    symbol: str = "BTCUSDT"
    feature_dir: Path = Path.home() / "Data" / "binance" / "features"
    kline_root: Path = Path.home() / "Data" / "binance"
    out_dir: Path = Path.home() / "CryptoDL" / "checkpoints"

    hidden: int = 32
    feat_drop_p: float = 0.15

    epochs: int = 40
    chunk_size: int = 192  # 2 days of 15-min bars (96/day)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10

    cost_bps: float = 2.0

    # loss: "sharpe" | "mv"
    loss: str = "sharpe"
    lambda_risk: float = 2.0  # only for loss="mv"


def checkpoint_suffix(cfg: TrainConfig) -> str:
    if cfg.loss == "sharpe":
        return "sharpe"
    return f"mv_l{cfg.lambda_risk:g}".replace(".", "p")


def build_loss_fn(cfg: TrainConfig) -> nn.Module:
    if cfg.loss == "sharpe":
        return NegSharpeLoss()
    if cfg.loss == "mv":
        return MeanVarianceLoss(lambda_risk=cfg.lambda_risk)
    raise ValueError(f"Unknown loss={cfg.loss!r}, use 'sharpe' or 'mv'")


def evaluate(
    model: MultiBranchLSTM,
    ds: CryptoDataset,
    cost_bps: float = 2.0,
    chunk_size: int = 192,
) -> dict:
    """Sequential playback evaluation at 15-min intervals."""
    model.eval()
    loader = DataLoader(ds, batch_size=chunk_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    all_net_pnl = []
    all_positions = []
    prev_pos = 0.0

    with torch.no_grad():
        for x8h, x1h, x1m, ret_bps, has_nan in loader:
            positions = model(x8h, x1h, x1m)

            gross_pnl = positions * ret_bps
            all_pos = torch.cat(
                [torch.tensor([prev_pos], dtype=positions.dtype), positions]
            )
            turnover = torch.abs(all_pos[1:] - all_pos[:-1])
            net_pnl = gross_pnl - cost_bps * turnover

            all_net_pnl.append(net_pnl)
            all_positions.append(positions)
            prev_pos = positions[-1].item()

    net_pnl = torch.cat(all_net_pnl)
    positions = torch.cat(all_positions)

    mean_pnl = net_pnl.mean().item()
    std_pnl = net_pnl.std(unbiased=True).item() if len(net_pnl) > 1 else 1e-8
    sharpe = (mean_pnl / max(std_pnl, 1e-8)) * np.sqrt(PERIODS_PER_YEAR)

    cum_pnl = net_pnl.cumsum(0)
    max_dd = (cum_pnl.cummax(0).values - cum_pnl).max().item()

    turnover_vals = torch.abs(positions[1:] - positions[:-1])

    return {
        "sharpe_annual": sharpe,
        "mean_pnl_bps": mean_pnl,
        "total_pnl_bps": net_pnl.sum().item(),
        "max_drawdown_bps": max_dd,
        "avg_turnover_per_rebal": turnover_vals.mean().item(),
        "daily_turnover": turnover_vals.mean().item() * 96,
        "avg_abs_position": positions.abs().mean().item(),
        "pct_long": (positions > 0.1).float().mean().item() * 100,
        "pct_short": (positions < -0.1).float().mean().item() * 100,
        "pct_flat": (positions.abs() < 0.1).float().mean().item() * 100,
        "n_periods": len(net_pnl),
    }


def train(
    model: MultiBranchLSTM,
    train_ds: CryptoDataset,
    val_ds: CryptoDataset,
    cfg: TrainConfig,
) -> float:
    ckpt_name = f"{cfg.symbol}_best_{checkpoint_suffix(cfg)}.pt"
    print("\n" + "=" * 60)
    if cfg.loss == "sharpe":
        print(f"Training: Neg-Sharpe, cost={cfg.cost_bps}bps ({len(train_ds)} train, {len(val_ds)} val)")
    else:
        print(
            f"Training: Mean–Variance λ={cfg.lambda_risk}, cost={cfg.cost_bps}bps "
            f"({len(train_ds)} train, {len(val_ds)} val)"
        )
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loader = DataLoader(train_ds, batch_size=cfg.chunk_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    loss_fn = build_loss_fn(cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_sharpe = -float("inf")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        epoch_chunks = 0
        prev_position = torch.zeros(1)

        for x8h, x1h, x1m, ret_bps, has_nan in loader:
            positions = model(x8h, x1h, x1m)
            loss = loss_fn(positions, ret_bps, prev_position, cfg.cost_bps)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            prev_position = positions[-1:].detach()
            epoch_loss += loss.item()
            epoch_chunks += 1

        scheduler.step()
        avg_loss = epoch_loss / max(epoch_chunks, 1)

        val_m = evaluate(model, val_ds, cfg.cost_bps, cfg.chunk_size)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch+1:2d}/{cfg.epochs}  "
            f"loss={avg_loss:+.4f}  "
            f"val_sharpe={val_m['sharpe_annual']:+.2f}  "
            f"val_pnl={val_m['total_pnl_bps']:+.0f}bps  "
            f"|pos|={val_m['avg_abs_position']:.2f}  "
            f"daily_turn={val_m['daily_turnover']:.1f}  "
            f"time={elapsed:.1f}s"
        )

        if val_m["sharpe_annual"] > best_val_sharpe:
            best_val_sharpe = val_m["sharpe_annual"]
            patience_counter = 0
            torch.save(model.state_dict(), cfg.out_dir / ckpt_name)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(
        torch.load(cfg.out_dir / ckpt_name, weights_only=True)
    )
    print(f"  Best val Sharpe: {best_val_sharpe:.2f}")
    return best_val_sharpe


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-dir", type=Path,
                        default=Path.home() / "Data" / "binance" / "features")
    parser.add_argument("--kline-root", type=Path,
                        default=Path.home() / "Data" / "binance")
    parser.add_argument("--out-dir", type=Path,
                        default=Path.home() / "CryptoDL" / "checkpoints")
    parser.add_argument(
        "--loss",
        choices=("sharpe", "mv"),
        default="sharpe",
        help="sharpe: −mean/std on net_pnl; mv: −(mean − λ·var)",
    )
    parser.add_argument(
        "--lambda-risk",
        type=float,
        default=2.0,
        help="Mean–variance risk aversion λ (only --loss mv)",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        symbol=args.symbol, hidden=args.hidden, epochs=args.epochs,
        chunk_size=args.chunk_size, lr=args.lr,
        feature_dir=args.feature_dir, kline_root=args.kline_root,
        out_dir=args.out_dir,
        loss=args.loss,
        lambda_risk=args.lambda_risk,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    store = load_store(cfg.feature_dir, cfg.kline_root, cfg.symbol)

    print("\nComputing rebalance indices (every 15 min)...")
    train_idx = compute_rebal_indices(store, "train")
    val_idx = compute_rebal_indices(store, "val")
    test_idx = compute_rebal_indices(store, "test")
    paper_idx = compute_rebal_indices(store, "paper")

    train_ds = CryptoDataset(store, train_idx)
    val_ds = CryptoDataset(store, val_idx)
    test_ds = CryptoDataset(store, test_idx)
    paper_ds = CryptoDataset(store, paper_idx)

    model = MultiBranchLSTM(f=F, hidden=cfg.hidden, feat_drop_p=cfg.feat_drop_p)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: MultiBranchLSTM (hidden={cfg.hidden}), {n_params:,} params")
    print(f"Rebalance interval: {REBAL_INTERVAL} min")
    print(f"Chunk size: {cfg.chunk_size} ({cfg.chunk_size * REBAL_INTERVAL / 60:.0f} hours)")

    # quick sanity: check initial output scale
    model.eval()
    with torch.no_grad():
        sample = train_ds[0]
        x8h, x1h, x1m = sample[0].unsqueeze(0), sample[1].unsqueeze(0), sample[2].unsqueeze(0)
        pos = model(x8h, x1h, x1m)
        print(f"Initial sample position: {pos.item():.4f}")

    best_sharpe = train(model, train_ds, val_ds, cfg)

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    test_m = evaluate(model, test_ds, cfg.cost_bps, cfg.chunk_size)
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Always-long baseline on test
    print("\nBaseline: Always Long (no cost)")
    test_rets = []
    for idx in test_idx:
        c0 = store.close_1m[idx]
        c1 = store.close_1m[idx + REBAL_INTERVAL]
        if c0 > 0:
            test_rets.append(np.log(c1 / c0) * 10000)
    test_rets = np.array(test_rets)
    long_sharpe = (test_rets.mean() / test_rets.std()) * np.sqrt(PERIODS_PER_YEAR)
    print(f"  Always-long Sharpe: {long_sharpe:.2f}")
    print(f"  Always-long total PnL: {test_rets.sum():.0f} bps")

    results = {
        "symbol": cfg.symbol,
        "hidden": cfg.hidden,
        "n_params": n_params,
        "loss": cfg.loss,
        "lambda_risk": cfg.lambda_risk if cfg.loss == "mv" else None,
        "best_val_sharpe": best_sharpe,
        "test": test_m,
        "always_long_sharpe": float(long_sharpe),
    }
    res_name = f"{cfg.symbol}_results_{checkpoint_suffix(cfg)}.json"
    with open(cfg.out_dir / res_name, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {cfg.out_dir / res_name}")


if __name__ == "__main__":
    main()
