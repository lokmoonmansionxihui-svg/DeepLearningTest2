#!/bin/bash
echo "Waiting for current runs to finish..."
while kill -0 261086 2>/dev/null; do sleep 10; done
while kill -0 261087 2>/dev/null; do sleep 10; done
echo "Current runs finished. Starting night queue..."

export CUDA_VISIBLE_DEVICES=0

echo "=== 1. LSTM + Time Features + Chunk Sharpe (Old Loss) ==="
uv run python -u scripts/train.py --model lstm --loss sharpe-chunk --variant original --hidden 32 --epochs 200 --seed 42 > logs_night_1_lstm_chunk.txt 2>&1

echo "=== 2. LSTM + Time Features + TwoPass Demean (New Loss) ==="
uv run python -u scripts/train.py --model lstm --loss sharpe --variant nobias --hidden 64 --epochs 200 --seed 42 --demean-window 100 > logs_night_2_lstm_twopass.txt 2>&1

echo "=== 3. Transformer + Time Features + Chunk Sharpe ==="
uv run python -u scripts/train.py --model transformer --loss sharpe-chunk --variant original --epochs 200 --seed 42 > logs_night_3_tf_chunk.txt 2>&1

echo "=== 4. Transformer + Time Features + TwoPass Demean ==="
uv run python -u scripts/train.py --model transformer --loss sharpe --variant nobias --epochs 200 --seed 42 --demean-window 100 > logs_night_4_tf_twopass.txt 2>&1

echo "=== 5. Transformer + Time Features + MV Loss (lambda=1.0) ==="
uv run python -u scripts/train.py --model transformer --loss mv --lambda-risk 1.0 --variant nobias --epochs 200 --seed 42 > logs_night_5_tf_mv.txt 2>&1

echo "All night jobs completed!"