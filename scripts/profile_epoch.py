#!/usr/bin/env python3
"""Profile one epoch to find the bottleneck."""
import time
import numpy as np
import torch
from pathlib import Path
from train import (
    load_store, compute_rebal_indices, CryptoDataset, MultiBranchLSTM,
    collate_fn, compute_rolling_mean_returns, _net_pnl_from_positions,
    set_seed, DEVICE, USE_AMP, F, REBAL_INTERVAL,
)
from torch.utils.data import DataLoader

set_seed(42)
print(f"Device: {DEVICE}")

t0 = time.time()
store = load_store(
    Path.home() / "Data" / "binance" / "features",
    Path.home() / "Data" / "binance",
    "BTCUSDT",
)
train_idx = compute_rebal_indices(store, "train")
train_ds = CryptoDataset(store, train_idx)
print(f"Data load: {time.time()-t0:.1f}s")

t0 = time.time()
loader = DataLoader(train_ds, batch_size=1920, shuffle=False,
                    collate_fn=collate_fn, num_workers=4,
                    pin_memory=True, persistent_workers=True)
print(f"DataLoader creation: {time.time()-t0:.3f}s")

model = MultiBranchLSTM(f=F, hidden=8, variant="nobias").to(DEVICE)
rolling_mean = compute_rolling_mean_returns(store, train_idx, 100)
scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

cost_bps = 2.0
eps_var = 1e-8
eps_sigma = 1e-6

# === Pass 1 ===
model.train()
t_pass1_total = time.time()
t_dataload = 0
t_transfer = 0
t_forward = 0
t_pnl = 0

all_pnl_cpu = []
rng_states = []
prev_pos = torch.zeros(1, device=DEVICE)
offset = 0
n_chunks = 0

t_iter_start = time.time()
with torch.no_grad():
    for x8h, x1h, x1m, ret_bps, _ in loader:
        t1 = time.time()
        t_dataload += t1 - t_iter_start

        x8h = x8h.to(DEVICE, non_blocking=True)
        x1h = x1h.to(DEVICE, non_blocking=True)
        x1m = x1m.to(DEVICE, non_blocking=True)
        ret_bps = ret_bps.to(DEVICE, non_blocking=True)
        torch.cuda.synchronize()
        t2 = time.time()
        t_transfer += t2 - t1

        rng_states.append(torch.cuda.get_rng_state())
        chunk_len = ret_bps.shape[0]
        ret_for_pnl = ret_bps - rolling_mean[offset:offset+chunk_len].to(DEVICE)

        with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
            pos = model(x8h, x1h, x1m)
        torch.cuda.synchronize()
        t3 = time.time()
        t_forward += t3 - t2

        pnl = _net_pnl_from_positions(pos.float(), ret_for_pnl, prev_pos, cost_bps)
        all_pnl_cpu.append(pnl.cpu())
        prev_pos = pos[-1:].float()
        offset += chunk_len
        t4 = time.time()
        t_pnl += t4 - t3

        n_chunks += 1
        t_iter_start = time.time()

t_pass1_elapsed = time.time() - t_pass1_total
print(f"\n=== Pass 1 ({n_chunks} chunks) total: {t_pass1_elapsed:.2f}s ===")
print(f"  DataLoader wait: {t_dataload:.2f}s ({t_dataload/t_pass1_elapsed*100:.0f}%)")
print(f"  CPU->GPU transfer: {t_transfer:.2f}s ({t_transfer/t_pass1_elapsed*100:.0f}%)")
print(f"  Model forward: {t_forward:.2f}s ({t_forward/t_pass1_elapsed*100:.0f}%)")
print(f"  PnL compute: {t_pnl:.2f}s ({t_pnl/t_pass1_elapsed*100:.0f}%)")

# === Stats ===
t_stats = time.time()
full_pnl = torch.cat(all_pnl_cpu)
N = full_pnl.numel()
mu = full_pnl.mean().item()
var = max(full_pnl.var(unbiased=False).item(), eps_var)
sigma = var ** 0.5
sigma_eps = sigma + eps_sigma
t_stats_elapsed = time.time() - t_stats
print(f"\nStats compute: {t_stats_elapsed:.4f}s")

# === Pass 2 ===
t_pass2_total = time.time()
t_dataload2 = 0
t_transfer2 = 0
t_forward2 = 0
t_backward2 = 0

optimizer.zero_grad()
prev_pos = torch.zeros(1, device=DEVICE)
offset = 0

t_iter_start = time.time()
for i, (x8h, x1h, x1m, ret_bps, _) in enumerate(loader):
    t1 = time.time()
    t_dataload2 += t1 - t_iter_start

    torch.cuda.set_rng_state(rng_states[i])
    x8h = x8h.to(DEVICE, non_blocking=True)
    x1h = x1h.to(DEVICE, non_blocking=True)
    x1m = x1m.to(DEVICE, non_blocking=True)
    ret_bps = ret_bps.to(DEVICE, non_blocking=True)
    torch.cuda.synchronize()
    t2 = time.time()
    t_transfer2 += t2 - t1

    chunk_len = ret_bps.shape[0]
    ret_for_pnl = ret_bps - rolling_mean[offset:offset+chunk_len].to(DEVICE)

    with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
        pos = model(x8h, x1h, x1m)
    pnl = _net_pnl_from_positions(pos.float(), ret_for_pnl, prev_pos, cost_bps)
    torch.cuda.synchronize()
    t3 = time.time()
    t_forward2 += t3 - t2

    R_chunk = full_pnl[offset:offset+pnl.numel()].to(DEVICE)
    grad = -(1.0 / (N * sigma_eps)) * (1.0 - mu * (R_chunk - mu) / (sigma_eps * sigma))
    scaler.scale(pnl).backward(gradient=grad)
    torch.cuda.synchronize()
    t4 = time.time()
    t_backward2 += t4 - t3

    prev_pos = pos[-1:].detach().float()
    offset += chunk_len
    t_iter_start = time.time()

t_pass2_elapsed = time.time() - t_pass2_total
print(f"\n=== Pass 2 ({n_chunks} chunks) total: {t_pass2_elapsed:.2f}s ===")
print(f"  DataLoader wait: {t_dataload2:.2f}s ({t_dataload2/t_pass2_elapsed*100:.0f}%)")
print(f"  CPU->GPU transfer: {t_transfer2:.2f}s ({t_transfer2/t_pass2_elapsed*100:.0f}%)")
print(f"  Model forward: {t_forward2:.2f}s ({t_forward2/t_pass2_elapsed*100:.0f}%)")
print(f"  Backward: {t_backward2:.2f}s ({t_backward2/t_pass2_elapsed*100:.0f}%)")

# === Optimizer step ===
t_opt = time.time()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
t_opt_elapsed = time.time() - t_opt
print(f"\nOptimizer step: {t_opt_elapsed:.4f}s")

print(f"\n=== TOTAL: {t_pass1_elapsed + t_stats_elapsed + t_pass2_elapsed + t_opt_elapsed:.2f}s ===")
