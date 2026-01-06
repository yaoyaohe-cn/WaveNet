#!/bin/bash

if [ ! -d "./logs/Optuna" ]; then
    mkdir -p ./logs/Optuna
fi

export CUDA_VISIBLE_DEVICES=0

dataset=Electricity
pred_len=96
n_trials=80

echo "Starting Comprehensive Tuning for ${dataset}..."

python -u run_SWN_optuna.py \
    --data $dataset \
    --pred_len $pred_len \
    --n_trials $n_trials \
    --train_epochs 30 \
    --patience 5 \
    > logs/Optuna/tuning_v4_${dataset}_${pred_len}.log

echo "Optimization Finished. Check logs/Optuna/tuning_${dataset}_${pred_len}.log"
