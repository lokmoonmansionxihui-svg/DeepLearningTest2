#!/bin/bash
cd /home/chenx/CryptoDL
COMMON="--model lstm --loss sharpe --variant nobias --hidden 16 --epochs 300 --seed 42 --demean-window 100 --chunk-size 1920"

echo "=== 1. ret only ==="
uv run python -u scripts/train.py $COMMON --keep-groups ret > logs_feat_ret.txt 2>&1

echo "=== 2. basis only ==="
uv run python -u scripts/train.py $COMMON --keep-groups basis > logs_feat_basis.txt 2>&1

echo "=== 3. imbalance only ==="
uv run python -u scripts/train.py $COMMON --keep-groups imbalance > logs_feat_imbalance.txt 2>&1

echo "=== 4. ret+vol+volume ==="
uv run python -u scripts/train.py $COMMON --keep-groups ret,vol,volume > logs_feat_ret_vol_volume.txt 2>&1

echo "=== 5. ret+basis ==="
uv run python -u scripts/train.py $COMMON --keep-groups ret,basis > logs_feat_ret_basis.txt 2>&1

echo "=== 6. ret+vol+volume+basis ==="
uv run python -u scripts/train.py $COMMON --keep-groups ret,vol,volume,basis > logs_feat_ret_vol_volume_basis.txt 2>&1

echo "=== 7. ret+vol+volume+imbalance ==="
uv run python -u scripts/train.py $COMMON --keep-groups ret,vol,volume,imbalance > logs_feat_ret_vol_volume_imbalance.txt 2>&1

echo "=== 8. all (baseline) ==="
uv run python -u scripts/train.py $COMMON > logs_feat_all.txt 2>&1

echo "All feature ablation jobs done!"
