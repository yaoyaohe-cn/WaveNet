#!/bin/bash

if [ ! -d "./logs/SWN" ]; then
    mkdir -p ./logs/SWN
fi

export CUDA_VISIBLE_DEVICES=0

model_name=SpectralWaveletNet
dataset=Weather

seq_lens=(512 512 512 512) 
pred_lens=(96 192 336 720)

wavelets=(db4 db4 db4 db4)
levels=(2 2 2 2) 
d_models=(96 96 96 96)
dropouts=(0.2 0.2 0.2 0.2)
decay=(1.584039331470833e-05 1.584039331470833e-05 1.584039331470833e-05 1.584039331470833e-05)
learning_rates=(0.0025 0.0025 0.0025 0.0025)
batches=(64 64 64 64)
epochs=(30 30 30 30)
patiences=(5 5 5 5)

for i in "${!pred_lens[@]}"; do
    log_file="logs/SWN/${dataset}_${pred_lens[$i]}.log"
    echo "Running ${dataset} Prediction ${pred_lens[$i]}..."
    python -u run_SWN.py \
        --model $model_name \
        --data $dataset \
        --seq_len ${seq_lens[$i]} \
        --pred_len ${pred_lens[$i]} \
        --d_model ${d_models[$i]} \
        --wavelet ${wavelets[$i]} \
        --level ${levels[$i]} \
        --dropout ${dropouts[$i]} \
        --weight_decay ${decay[$i]} \
        --learning_rate ${learning_rates[$i]} \
        --batch_size ${batches[$i]} \
        --train_epochs ${epochs[$i]} \
        --patience ${patiences[$i]} \
        > $log_file
done
