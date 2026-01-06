#!/bin/bash

# Create logs directory
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./logs/SWN" ]; then
    mkdir ./logs/SWN
fi

export CUDA_VISIBLE_DEVICES=0

model_name=SpectralWaveletNet
dataset=ETTh2

# Hyperparameters
seq_lens=(512 512 512 512) 
pred_lens=(96 192 336 720)
wavelets=(db4 db4 db4 db4)
levels=(3 3 3 3) 
d_models=(128 128 128 128)
dropouts=(0.2 0.2 0.2 0.2)
decay=(0.00047377723860348293 0.00047377723860348293 0.00047377723860348293 0.00047377723860348293)
learning_rates=(0.0025 0.0025 0.0025 0.0025)
batches=(256 256 256 256)
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
        --learning_rate ${learning_rates[$i]} \
        --batch_size ${batches[$i]} \
        --weight_decay ${decay[$i]} \
        --train_epochs ${epochs[$i]} \
        --patience ${patiences[$i]} \
        > $log_file
done
