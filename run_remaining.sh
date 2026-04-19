#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd /home/chenx/CryptoDL

echo "=== 1. LSTM + Time Features + TwoPass Demean ==="
uv run python -u scripts/train.py --model lstm --loss sharpe --variant nobias --hidden 64 --epochs 200 --seed 42 --demean-window 100 > logs_night_2_lstm_twopass.txt 2>&1

echo "=== 2. Transformer + Time Features + TwoPass Demean ==="
uv run python -u scripts/train.py --model transformer --loss sharpe --variant nobias --epochs 200 --seed 42 --demean-window 100 > logs_night_4_tf_twopass.txt 2>&1

echo "All jobs completed!"
