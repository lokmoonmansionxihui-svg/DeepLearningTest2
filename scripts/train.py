#!/usr/bin/env python3
"""Training: supports single-phase (Sharpe/MV) and two-phase (Huber pretrain + MV finetune).

Every 15 minutes:
  - Multi-resolution features → position = tanh(signal) ∈ [-1, 1]
  - Gross PnL = position × 15min_return (bps)
  - Cost = cost_bps × |Δposition| at rebalance boundaries only
  - Net PnL = gross − cost

Loss modes (see --loss):
  - sharpe:       Two-Pass full-sequence Sharpe (DeePM Appendix C, exact gradient).
                  Uses --demean-window (default 100d) rolling mean subtraction to remove
                  unconditional drift and force the model to learn timing signals.
                  Evaluation always uses raw returns.
  - sharpe-chunk: Per-chunk Sharpe (original, −mean/std on chunk_size window)
  - mv:           minimize −(mean(net_pnl) − λ·var(net_pnl))

Usage:
    uv run python scripts/train.py --symbol BTCUSDT --loss sharpe
    uv run python scripts/train.py --symbol BTCUSDT --loss sharpe-chunk
    uv run python scripts/train.py --symbol BTCUSDT --loss mv --lambda-risk 1.0
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

F = 27  # updated dynamically if --drop-basis
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

        def get_time_features(ts_array):
            mod = (ts_array % 86400) / 60.0
            dow = ((ts_array // 86400) + 4) % 7
            return np.column_stack([
                np.sin(2 * np.pi * mod / 1440.0),
                np.cos(2 * np.pi * mod / 1440.0),
                np.sin(2 * np.pi * dow / 7.0),
                np.cos(2 * np.pi * dow / 7.0)
            ]).astype(np.float32)

        tf_1m = get_time_features(self.s.ts_1m[i - LOOKBACK_1M + 1: i + 1])
        x_1m = np.concatenate([x_1m, tf_1m], axis=1)

        tf_1h = get_time_features(self.s.ts_1h[h - LOOKBACK_1H + 1: h + 1])
        x_1h = np.concatenate([x_1h, tf_1h], axis=1)

        tf_8h = get_time_features(self.s.ts_8h[e - LOOKBACK_8H + 1: e + 1])
        x_8h = np.concatenate([x_8h, tf_8h], axis=1)

        c_now = self.s.close_1m[i]
        c_fwd = self.s.close_1m[i + REBAL_INTERVAL]
        fwd_ret = np.log(c_fwd / c_now) * 10000 if c_now > 0 else 0.0  # bps

        has_nan = bool(
            np.isnan(x_1m).any() or np.isnan(x_1h).any() or np.isnan(x_8h).any()
        )

        return (
            torch.from_numpy(x_8h.copy()),
            torch.from_numpy(x_1h.copy()),
            torch.from_numpy(x_1m.copy()),
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


@dataclass
class GPUData:
    """All samples preloaded on GPU for zero-overhead iteration."""
    x_8h: torch.Tensor   # (N, LOOKBACK_8H, F)
    x_1h: torch.Tensor   # (N, LOOKBACK_1H, F)
    x_1m: torch.Tensor   # (N, LOOKBACK_1M, F)
    ret_bps: torch.Tensor  # (N,)
    n: int

    def chunks(self, chunk_size: int):
        """Yield (x_8h, x_1h, x_1m, ret_bps) chunks."""
        for start in range(0, self.n, chunk_size):
            end = min(start + chunk_size, self.n)
            yield (
                self.x_8h[start:end],
                self.x_1h[start:end],
                self.x_1m[start:end],
                self.ret_bps[start:end],
            )


def _time_features_np(ts: np.ndarray) -> np.ndarray:
    mod = (ts % 86400) / 60.0
    dow = ((ts // 86400) + 4) % 7
    return np.column_stack([
        np.sin(2 * np.pi * mod / 1440.0),
        np.cos(2 * np.pi * mod / 1440.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
    ]).astype(np.float32)


def preload_to_gpu(store: TimeSeriesStore, indices: np.ndarray, label: str = "",
                   drop_basis: bool = False, keep_groups: str = "") -> GPUData:
    """Precompute all samples and move to GPU in one shot."""
    t0 = time.time()
    N = len(indices)

    tf_1m_all = _time_features_np(store.ts_1m)
    tf_1h_all = _time_features_np(store.ts_1h)
    tf_8h_all = _time_features_np(store.ts_8h)

    FEAT_GROUPS = {
        "ret": [0,1,2,3,4],
        "vol": [5],
        "volume": [6],
        "imbalance": [7,8,9,10,11],
        "basis": [12,13,14,15,16],
        "voldiff": [17],
        "imbdiff": [18,19,20,21,22],
    }
    f1m, f1h, f8h = store.feat_1m, store.feat_1h, store.feat_8h
    if keep_groups:
        keep = []
        for g in keep_groups.split(","):
            keep.extend(FEAT_GROUPS[g.strip()])
        keep.sort()
        f1m, f1h, f8h = f1m[:, keep], f1h[:, keep], f8h[:, keep]
    elif drop_basis:
        keep = [i for i in range(f1m.shape[1]) if i not in FEAT_GROUPS["basis"]]
        f1m, f1h, f8h = f1m[:, keep], f1h[:, keep], f8h[:, keep]
    feat_1m_full = np.concatenate([f1m, tf_1m_all], axis=1)
    feat_1h_full = np.concatenate([f1h, tf_1h_all], axis=1)
    feat_8h_full = np.concatenate([f8h, tf_8h_all], axis=1)

    f_total = feat_1m_full.shape[1]

    x8h = np.zeros((N, LOOKBACK_8H, f_total), dtype=np.float32)
    x1h = np.zeros((N, LOOKBACK_1H, f_total), dtype=np.float32)
    x1m = np.zeros((N, LOOKBACK_1M, f_total), dtype=np.float32)
    rets = np.zeros(N, dtype=np.float32)

    close = store.close_1m
    for k, i in enumerate(indices):
        x1m[k] = feat_1m_full[i - LOOKBACK_1M + 1: i + 1]
        h = store.h_idx[i]
        x1h[k] = feat_1h_full[h - LOOKBACK_1H + 1: h + 1]
        e = store.e_idx[i]
        x8h[k] = feat_8h_full[e - LOOKBACK_8H + 1: e + 1]
        c0, c1 = close[i], close[i + REBAL_INTERVAL]
        rets[k] = np.log(c1 / c0) * 10000 if c0 > 0 else 0.0

    np.nan_to_num(x8h, copy=False)
    np.nan_to_num(x1h, copy=False)
    np.nan_to_num(x1m, copy=False)

    gpu_data = GPUData(
        x_8h=torch.from_numpy(x8h).to(DEVICE),
        x_1h=torch.from_numpy(x1h).to(DEVICE),
        x_1m=torch.from_numpy(x1m).to(DEVICE),
        ret_bps=torch.from_numpy(rets).to(DEVICE),
        n=N,
    )
    elapsed = time.time() - t0
    mem_gb = (x8h.nbytes + x1h.nbytes + x1m.nbytes + rets.nbytes) / 1e9
    print(f"  Preloaded {label}: {N:,} samples, {mem_gb:.2f} GB → GPU in {elapsed:.1f}s", flush=True)
    return gpu_data


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


VARIANTS = ("original", "nobias")

def _build_head(variant: str, concat_dim: int, hidden: int) -> nn.Sequential:
    if variant == "original":
        return nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    # nobias
    return nn.Sequential(
        nn.Linear(concat_dim, hidden, bias=False),
        nn.Tanh(),
        nn.Dropout(0.3),
        nn.Linear(hidden, 1, bias=False),
    )


class MultiBranchLSTM(nn.Module):
    def __init__(self, f: int = F, hidden: int = 64, feat_drop_p: float = 0.15, variant: str = "nobias"):
        super().__init__()
        self.variant = variant
        self.feat_drop = FeatureDropout(p=feat_drop_p)

        use_bias = (variant == "original")
        self.lstm_8h = nn.LSTM(f, hidden, num_layers=1, batch_first=True, bias=use_bias)
        self.lstm_1h = nn.LSTM(f, hidden, num_layers=1, batch_first=True, bias=use_bias)
        self.lstm_1m = nn.LSTM(f, hidden, num_layers=1, batch_first=True, bias=use_bias)

        concat_dim = hidden * 3
        self.head = _build_head(variant, concat_dim, hidden)

        nn.init.xavier_normal_(self.head[-1].weight, gain=0.5)

    def forward(self, x_8h, x_1h, x_1m, raw: bool = False) -> torch.Tensor:
        """Returns position in [-1, 1] (raw=False) or unbounded signal (raw=True)."""
        x_8h = torch.nan_to_num(self.feat_drop(x_8h), 0.0)
        x_1h = torch.nan_to_num(self.feat_drop(x_1h), 0.0)
        x_1m = torch.nan_to_num(self.feat_drop(x_1m), 0.0)

        _, (h_8h, _) = self.lstm_8h(x_8h)
        _, (h_1h, _) = self.lstm_1h(x_1h)
        _, (h_1m, _) = self.lstm_1m(x_1m)

        combined = torch.cat([h_8h[-1], h_1h[-1], h_1m[-1]], dim=1)
        signal = self.head(combined).squeeze(-1)
        if raw:
            return signal
        return torch.tanh(signal)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class BranchTransformer(nn.Module):
    def __init__(self, f, d_model, nhead, d_ff, layers, dropout, max_len):
        super().__init__()
        self.proj = nn.Linear(f, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_enc(x)
        out = self.transformer(x)
        return out[:, 0, :]  # Return CLS token


class MultiBranchTransformer(nn.Module):
    def __init__(self, f: int = F, hidden: int = 64, feat_drop_p: float = 0.15, variant: str = "nobias"):
        super().__init__()
        self.variant = variant
        self.feat_drop = FeatureDropout(p=feat_drop_p)

        nhead = 4
        d_ff = hidden * 4
        layers = 2
        dropout = 0.3

        self.branch_8h = BranchTransformer(f, hidden, nhead, d_ff, layers, dropout, LOOKBACK_8H)
        self.branch_1h = BranchTransformer(f, hidden, nhead, d_ff, layers, dropout, LOOKBACK_1H)
        self.branch_1m = BranchTransformer(f, hidden, nhead, d_ff, layers, dropout, LOOKBACK_1M)

        concat_dim = hidden * 3
        self.head = _build_head(variant, concat_dim, hidden)

        nn.init.xavier_normal_(self.head[-1].weight, gain=0.5)

    def forward(self, x_8h, x_1h, x_1m, raw: bool = False) -> torch.Tensor:
        x_8h = torch.nan_to_num(self.feat_drop(x_8h), 0.0)
        x_1h = torch.nan_to_num(self.feat_drop(x_1h), 0.0)
        x_1m = torch.nan_to_num(self.feat_drop(x_1m), 0.0)

        h_8h = self.branch_8h(x_8h)
        h_1h = self.branch_1h(x_1h)
        h_1m = self.branch_1m(x_1m)

        combined = torch.cat([h_8h, h_1h, h_1m], dim=1)
        signal = self.head(combined).squeeze(-1)
        if raw:
            return signal
        return torch.tanh(signal)


# ── Loss ──────────────────────────────────────────────────────────────────────


def _net_pnl_from_positions(
    positions: torch.Tensor,
    fwd_ret_bps: torch.Tensor,
    prev_position: torch.Tensor,
    cost_coeff: float,
) -> torch.Tensor:
    gross_pnl = positions * fwd_ret_bps
    all_pos = torch.cat([prev_position, positions])
    turnover = torch.abs(all_pos[1:] - all_pos[:-1])
    return gross_pnl - cost_coeff * turnover


def compute_rolling_mean_returns(
    store: TimeSeriesStore, indices: np.ndarray, window_days: int = 100,
) -> torch.Tensor:
    """Causal rolling mean of forward 15-min returns (no look-ahead).

    Returns a tensor of shape (len(indices),) containing the rolling mean
    at each rebalance point. Window = window_days * 96 bars/day.
    """
    window = window_days * 96
    close = store.close_1m
    c_now = close[indices]
    c_fwd = close[indices + REBAL_INTERVAL]
    fwd_rets = np.where(c_now > 0, np.log(c_fwd / c_now) * 10000, 0.0).astype(np.float32)
    rolling_mean = pd.Series(fwd_rets).rolling(window, min_periods=1).mean().values
    return torch.tensor(rolling_mean, dtype=torch.float32)


class NegSharpeLoss(nn.Module):
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
        if net_pnl.numel() < 2:
            return -net_pnl.mean() / (self.eps + 1e-8)
        std = net_pnl.std(unbiased=True)
        return -net_pnl.mean() / (std + self.eps)


class MeanVarianceLoss(nn.Module):
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
    model: str = "lstm"
    feature_dir: Path = Path.home() / "Data" / "binance" / "features"
    kline_root: Path = Path.home() / "Data" / "binance"
    out_dir: Path = Path.home() / "CryptoDL" / "checkpoints"

    hidden: int = 64
    feat_drop_p: float = 0.15

    epochs: int = 200
    chunk_size: int = 192
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 30

    cost_bps: float = 2.0

    loss: str = "sharpe"
    lambda_risk: float = 2.0
    seed: int = 42
    variant: str = "nobias"
    demean_window: int = 100
    drop_basis: bool = False
    keep_groups: str = ""

    phase1_epochs: int = 0
    phase2_lr: float = 1e-4


def checkpoint_suffix(cfg: TrainConfig) -> str:
    parts = [cfg.model, cfg.variant]
    if cfg.loss == "sharpe":
        parts.append("sharpe_fullseq")
        if cfg.demean_window > 0:
            parts.append(f"dm{cfg.demean_window}d")
    elif cfg.loss == "sharpe-chunk":
        parts.append("sharpe_chunk")
    else:
        parts.append(f"mv_l{cfg.lambda_risk:g}".replace(".", "p"))
    if cfg.keep_groups:
        parts.append(f"feat_{cfg.keep_groups.replace(',', '_')}")
    elif cfg.drop_basis:
        parts.append("nobasis")
    parts.append(f"s{cfg.seed}")
    return "_".join(parts)


def build_loss_fn(cfg: TrainConfig) -> nn.Module:
    if cfg.loss in ("sharpe", "sharpe-chunk"):
        return NegSharpeLoss()
    if cfg.loss == "mv":
        return MeanVarianceLoss(lambda_risk=cfg.lambda_risk)
    raise ValueError(f"Unknown loss={cfg.loss!r}, use 'sharpe', 'sharpe-chunk', or 'mv'")


def compute_label_stats(store: TimeSeriesStore, indices: np.ndarray) -> tuple[float, float]:
    """Compute mean/std of forward log returns (bps) on given indices."""
    rets = []
    for i in indices:
        c0 = store.close_1m[i]
        c1 = store.close_1m[i + REBAL_INTERVAL]
        if c0 > 0:
            rets.append(np.log(c1 / c0) * 10000)
    rets = np.array(rets, dtype=np.float64)
    return float(rets.mean()), float(max(rets.std(), 1e-8))


def train_phase1(
    model: nn.Module,
    train_ds: CryptoDataset,
    val_ds: CryptoDataset,
    cfg: TrainConfig,
    label_mean: float,
    label_std: float,
) -> None:
    """Phase 1: Huber loss on z-scored forward returns. Shuffled, no tanh."""
    ckpt_name = f"{cfg.symbol}_phase1_best.pt"
    print("\n" + "=" * 60)
    print(f"Phase 1: Huber pretrain ({cfg.phase1_epochs} epochs, LR={cfg.lr})")
    print(f"  Label z-score: mean={label_mean:.4f}, std={label_std:.4f}")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loader = DataLoader(train_ds, batch_size=cfg.chunk_size, shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    loss_fn = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.phase1_epochs
    )
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.phase1_epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for x8h, x1h, x1m, ret_bps, has_nan in loader:
            x8h = x8h.to(DEVICE, non_blocking=True)
            x1h = x1h.to(DEVICE, non_blocking=True)
            x1m = x1m.to(DEVICE, non_blocking=True)
            ret_bps = ret_bps.to(DEVICE, non_blocking=True)

            z_label = (ret_bps - label_mean) / label_std

            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                raw_signal = model(x8h, x1h, x1m, raw=True)
                loss = loss_fn(raw_signal, z_label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate with Huber loss
        model.eval()
        val_loader = DataLoader(val_ds, batch_size=cfg.chunk_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for x8h, x1h, x1m, ret_bps, has_nan in val_loader:
                x8h = x8h.to(DEVICE, non_blocking=True)
                x1h = x1h.to(DEVICE, non_blocking=True)
                x1m = x1m.to(DEVICE, non_blocking=True)
                ret_bps = ret_bps.to(DEVICE, non_blocking=True)
                z_label = (ret_bps - label_mean) / label_std
                with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                    raw_signal = model(x8h, x1h, x1m, raw=True)
                val_loss += loss_fn(raw_signal, z_label).item()
                val_n += 1

        avg_val_loss = val_loss / max(val_n, 1)
        elapsed = time.time() - t0

        print(
            f"  P1 Epoch {epoch+1:2d}/{cfg.phase1_epochs}  "
            f"train_loss={avg_loss:.4f}  val_loss={avg_val_loss:.4f}  "
            f"time={elapsed:.1f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.out_dir / ckpt_name)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Phase 1 early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(
        torch.load(cfg.out_dir / ckpt_name, weights_only=True, map_location=DEVICE)
    )
    print(f"  Phase 1 best val loss: {best_val_loss:.4f}")


def evaluate(
    model: nn.Module,
    gpu_data: GPUData,
    cost_bps: float = 2.0,
    chunk_size: int = 1920,
) -> dict:
    """Sequential playback evaluation at 15-min intervals."""
    model.eval()

    all_net_pnl: list[torch.Tensor] = []
    all_positions: list[torch.Tensor] = []
    prev_pos = torch.zeros(1, device=DEVICE)

    with torch.no_grad():
        for x8h, x1h, x1m, ret_bps in gpu_data.chunks(chunk_size):
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                positions = model(x8h, x1h, x1m)

            positions = positions.float()

            gross_pnl = positions * ret_bps
            all_pos = torch.cat([prev_pos, positions])
            turnover = torch.abs(all_pos[1:] - all_pos[:-1])
            net_pnl = gross_pnl - cost_bps * turnover

            all_net_pnl.append(net_pnl)
            all_positions.append(positions)
            prev_pos = positions[-1:].detach()

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


def train_epoch_fullseq_sharpe(
    model: nn.Module,
    gpu_data: GPUData,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
    rolling_mean: torch.Tensor | None = None,
) -> float:
    """Full-sequence Sharpe with two-pass exact gradient accumulation (DeePM Appendix C).

    Pass 1: forward all chunks without autograd, collect pnl to compute global mu/sigma.
    Pass 2: forward again with autograd (same RNG → same dropout), inject analytical
             gradient dL/dR_t and let PyTorch backprop through model parameters.
    """
    model.train()
    eps_var = 1e-8
    eps_sigma = 1e-6
    chunk_size = cfg.chunk_size

    # ── Pass 1: collect global statistics (no autograd) ──
    all_pnl: list[torch.Tensor] = []
    rng_states: list[torch.Tensor] = []
    prev_pos = torch.zeros(1, device=DEVICE)
    offset = 0

    with torch.no_grad():
        for x8h, x1h, x1m, ret_bps in gpu_data.chunks(chunk_size):
            rng_states.append(torch.cuda.get_rng_state())
            chunk_len = ret_bps.shape[0]

            if rolling_mean is not None:
                ret_for_pnl = ret_bps - rolling_mean[offset:offset + chunk_len]
            else:
                ret_for_pnl = ret_bps

            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                pos = model(x8h, x1h, x1m)
            pnl = _net_pnl_from_positions(pos.float(), ret_for_pnl, prev_pos, cfg.cost_bps)
            all_pnl.append(pnl)
            prev_pos = pos[-1:].float()
            offset += chunk_len

    full_pnl = torch.cat(all_pnl)
    N = full_pnl.numel()
    mu = full_pnl.mean().item()
    var = max(full_pnl.var(unbiased=False).item(), eps_var)
    sigma = var ** 0.5
    sigma_eps = sigma + eps_sigma

    # ── Pass 2: inject analytical gradient (DeePM Eq. 60) ──
    optimizer.zero_grad()
    prev_pos = torch.zeros(1, device=DEVICE)
    offset = 0

    for i, (x8h, x1h, x1m, ret_bps) in enumerate(gpu_data.chunks(chunk_size)):
        torch.cuda.set_rng_state(rng_states[i])
        chunk_len = ret_bps.shape[0]

        if rolling_mean is not None:
            ret_for_pnl = ret_bps - rolling_mean[offset:offset + chunk_len]
        else:
            ret_for_pnl = ret_bps

        with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
            pos = model(x8h, x1h, x1m)
        pnl = _net_pnl_from_positions(pos.float(), ret_for_pnl, prev_pos, cfg.cost_bps)

        R_chunk = full_pnl[offset:offset + pnl.numel()]
        grad = -(1.0 / (N * sigma_eps)) * (1.0 - mu * (R_chunk - mu) / (sigma_eps * sigma))

        scaler.scale(pnl).backward(gradient=grad)
        prev_pos = pos[-1:].detach().float()
        offset += chunk_len

    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    return -mu / sigma


def train(
    model: nn.Module,
    train_gpu: GPUData,
    val_gpu: GPUData,
    cfg: TrainConfig,
    store: TimeSeriesStore | None = None,
    train_indices: np.ndarray | None = None,
) -> float:
    use_twopass = (cfg.loss == "sharpe")
    ckpt_name = f"{cfg.symbol}_best_{checkpoint_suffix(cfg)}.pt"
    print("\n" + "=" * 60)
    if use_twopass:
        demean_tag = f", demean={cfg.demean_window}d" if cfg.demean_window > 0 else ""
        print(f"Training: Two-Pass Full-Sequence Sharpe, cost={cfg.cost_bps}bps{demean_tag} "
              f"({train_gpu.n} train, {val_gpu.n} val)")
    elif cfg.loss == "sharpe-chunk":
        print(f"Training: Per-Chunk Sharpe (chunk={cfg.chunk_size}), cost={cfg.cost_bps}bps "
              f"({train_gpu.n} train, {val_gpu.n} val)")
    else:
        print(
            f"Training: Mean–Variance λ={cfg.lambda_risk}, cost={cfg.cost_bps}bps "
            f"({train_gpu.n} train, {val_gpu.n} val)"
        )
    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print("=" * 60)

    rolling_mean: torch.Tensor | None = None
    if use_twopass and cfg.demean_window > 0 and store is not None and train_indices is not None:
        rolling_mean = compute_rolling_mean_returns(
            store, train_indices, cfg.demean_window
        ).to(DEVICE)
        print(f"  Return demeaning: {cfg.demean_window}-day rolling mean, "
              f"raw_ret_mean={rolling_mean.mean():+.4f}bps", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = build_loss_fn(cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)

    best_val_sharpe = -float("inf")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        t0 = time.time()

        if use_twopass:
            avg_loss = train_epoch_fullseq_sharpe(
                model, train_gpu, optimizer, scaler, cfg, rolling_mean,
            )
            scheduler.step()
        else:
            model.train()
            epoch_loss = 0.0
            epoch_chunks = 0
            prev_position = torch.zeros(1, device=DEVICE)

            for x8h, x1h, x1m, ret_bps in train_gpu.chunks(cfg.chunk_size):
                with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                    positions = model(x8h, x1h, x1m)
                    loss = loss_fn(positions.float(), ret_bps, prev_position, cfg.cost_bps)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                prev_position = positions[-1:].detach().float()
                epoch_loss += loss.item()
                epoch_chunks += 1

            scheduler.step()
            avg_loss = epoch_loss / max(epoch_chunks, 1)

        if use_twopass or cfg.loss == "sharpe-chunk":
            train_sharpe_ann = -avg_loss * np.sqrt(PERIODS_PER_YEAR)
            loss_str = f"train_sharpe={train_sharpe_ann:+.2f}"
        else:
            loss_str = f"loss={avg_loss:+.4f}"

        val_m = evaluate(model, val_gpu, cfg.cost_bps, cfg.chunk_size)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch+1:2d}/{cfg.epochs}  "
            f"{loss_str}  "
            f"val_sharpe={val_m['sharpe_annual']:+.2f}  "
            f"val_pnl={val_m['total_pnl_bps']:+.0f}bps  "
            f"|pos|={val_m['avg_abs_position']:.2f}  "
            f"daily_turn={val_m['daily_turnover']:.1f}  "
            f"time={elapsed:.1f}s",
            flush=True,
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
        torch.load(cfg.out_dir / ckpt_name, weights_only=True, map_location=DEVICE)
    )
    print(f"  Best val Sharpe: {best_val_sharpe:.2f}")
    return best_val_sharpe


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--model", choices=("lstm", "transformer"), default="lstm")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dir", type=Path,
                        default=Path.home() / "Data" / "binance" / "features")
    parser.add_argument("--kline-root", type=Path,
                        default=Path.home() / "Data" / "binance")
    parser.add_argument("--out-dir", type=Path,
                        default=Path.home() / "CryptoDL" / "checkpoints")
    parser.add_argument(
        "--loss",
        choices=("sharpe", "sharpe-chunk", "mv"),
        default="sharpe",
        help="sharpe: Two-Pass full-sequence; sharpe-chunk: per-chunk; mv: mean-variance",
    )
    parser.add_argument(
        "--lambda-risk",
        type=float,
        default=2.0,
        help="Mean–variance risk aversion λ (only --loss mv)",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="nobias",
        help="Head architecture variant: original, nobias",
    )
    parser.add_argument(
        "--demean-window", type=int, default=100,
        help="Rolling mean window in days for return demeaning in Sharpe loss (0=off)",
    )
    parser.add_argument(
        "--drop-basis", action="store_true",
        help="Remove basis_zscore features (columns 12-16)",
    )
    parser.add_argument(
        "--keep-groups", type=str, default="",
        help="Comma-separated feature groups to keep: ret,vol,volume,imbalance,basis,voldiff,imbdiff",
    )
    parser.add_argument(
        "--phase1-epochs", type=int, default=0,
        help="Phase 1 Huber pretrain epochs (0 = skip, single-phase only)",
    )
    parser.add_argument(
        "--phase2-lr", type=float, default=1e-4,
        help="Phase 2 learning rate (lower to preserve Phase 1 features)",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        symbol=args.symbol, model=args.model, hidden=args.hidden, epochs=args.epochs,
        chunk_size=args.chunk_size, lr=args.lr, seed=args.seed,
        feature_dir=args.feature_dir, kline_root=args.kline_root,
        out_dir=args.out_dir,
        loss=args.loss,
        lambda_risk=args.lambda_risk,
        variant=args.variant,
        demean_window=args.demean_window,
        drop_basis=args.drop_basis,
        keep_groups=args.keep_groups,
        phase1_epochs=args.phase1_epochs,
        phase2_lr=args.phase2_lr,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    print(f"Seed: {cfg.seed}, Device: {DEVICE}")

    store = load_store(cfg.feature_dir, cfg.kline_root, cfg.symbol)

    print("\nComputing rebalance indices (every 15 min)...")
    train_idx = compute_rebal_indices(store, "train")
    val_idx = compute_rebal_indices(store, "val")
    test_idx = compute_rebal_indices(store, "test")
    paper_idx = compute_rebal_indices(store, "paper")

    print("\nPreloading all data to GPU...")
    train_gpu = preload_to_gpu(store, train_idx, "train", drop_basis=cfg.drop_basis, keep_groups=cfg.keep_groups)
    val_gpu = preload_to_gpu(store, val_idx, "val", drop_basis=cfg.drop_basis, keep_groups=cfg.keep_groups)
    test_gpu = preload_to_gpu(store, test_idx, "test", drop_basis=cfg.drop_basis, keep_groups=cfg.keep_groups)
    paper_gpu = preload_to_gpu(store, paper_idx, "paper", drop_basis=cfg.drop_basis, keep_groups=cfg.keep_groups)

    f_actual = train_gpu.x_1m.shape[2]
    if cfg.drop_basis:
        print(f"  Dropped basis features → {f_actual} features per timestep")

    if cfg.model == "transformer":
        model = MultiBranchTransformer(f=f_actual, hidden=cfg.hidden, feat_drop_p=cfg.feat_drop_p, variant=cfg.variant)
    else:
        model = MultiBranchLSTM(f=f_actual, hidden=cfg.hidden, feat_drop_p=cfg.feat_drop_p, variant=cfg.variant)
    
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model.__class__.__name__} (hidden={cfg.hidden}, variant={cfg.variant}), {n_params:,} params")
    print(f"Rebalance interval: {REBAL_INTERVAL} min")
    print(f"Chunk size: {cfg.chunk_size} ({cfg.chunk_size * REBAL_INTERVAL / 60:.0f} hours)")

    model.eval()
    with torch.no_grad():
        x8h = train_gpu.x_8h[:1]
        x1h = train_gpu.x_1h[:1]
        x1m = train_gpu.x_1m[:1]
        pos = model(x8h, x1h, x1m)
        print(f"Initial sample position: {pos.item():.4f}")

    best_sharpe = train(model, train_gpu, val_gpu, cfg, store, train_idx)

    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    test_m = evaluate(model, test_gpu, cfg.cost_bps, cfg.chunk_size)
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

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
        "model": cfg.model,
        "variant": cfg.variant,
        "hidden": cfg.hidden,
        "n_params": n_params,
        "loss": cfg.loss,
        "lambda_risk": cfg.lambda_risk if cfg.loss == "mv" else None,
        "seed": cfg.seed,
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
