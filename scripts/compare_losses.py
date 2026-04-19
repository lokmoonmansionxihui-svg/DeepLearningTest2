#!/usr/bin/env python3
"""Run train.py for Sharpe vs several Mean–Variance λ values; aggregate JSON.

Example:
    uv run python scripts/compare_losses.py --epochs 22 --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--epochs", type=int, default=22)
    p.add_argument("--chunk-size", type=int, default=192)
    p.add_argument(
        "--mv-lambdas",
        default="0.5,1.0,2.0,4.0",
        help="comma-separated λ for --loss mv",
    )
    p.add_argument("--skip-sharpe", action="store_true")
    args = p.parse_args()

    out_dir = Path.home() / "CryptoDL" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    lambdas = [float(x.strip()) for x in args.mv_lambdas.split(",") if x.strip()]
    runs: list[tuple[str, list[str]]] = []
    if not args.skip_sharpe:
        runs.append(("sharpe", ["--loss", "sharpe"]))
    for lam in lambdas:
        runs.append((f"mv_{lam}", ["--loss", "mv", "--lambda-risk", str(lam)]))

    py = sys.executable
    train_py = ROOT / "scripts" / "train.py"
    base = [
        py,
        str(train_py),
        "--symbol",
        args.symbol,
        "--epochs",
        str(args.epochs),
        "--chunk-size",
        str(args.chunk_size),
    ]

    summary: list[dict] = []
    for name, extra in runs:
        cmd = base + extra
        print("\n" + "=" * 60)
        print("RUN:", " ".join(cmd))
        print("=" * 60)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"FAILED {name} exit={r.returncode}")
            continue
        # load written results json
        suffix = "sharpe" if extra[1] == "sharpe" else f"mv_l{float(extra[3]):g}".replace(".", "p")
        jpath = out_dir / f"{args.symbol}_results_{suffix}.json"
        if jpath.exists():
            with open(jpath) as f:
                row = json.load(f)
            row["run_name"] = name
            summary.append(row)
            print(f"  -> loaded {jpath}")

    out_json = out_dir / f"{args.symbol}_loss_comparison.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote comparison table to {out_json}")

    # compact table
    print("\n--- val Sharpe (best) | test Sharpe | test PnL bps | loss ---")
    for row in summary:
        loss = row.get("loss", "?")
        lam = row.get("lambda_risk")
        tag = f"{loss}" + (f" λ={lam}" if lam is not None else "")
        ts = row.get("test", {})
        print(
            f"  {row['best_val_sharpe']:+.2f}  |  {ts.get('sharpe_annual', 0):+.2f}  |  "
            f"{ts.get('total_pnl_bps', 0):+.0f}  |  {tag}"
        )


if __name__ == "__main__":
    main()
