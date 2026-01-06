#!/bin/bash

if [ ! -d "./logs/SWN" ]; then
    mkdir -p ./logs/SWN
fi

export CUDA_VISIBLE_DEVICES=0

model_name=SpectralWaveletNet
dataset=ETTm1

# For 15-min data, 336 covers 3.5 days. 
seq_lens=(336 336 336 336) 
pred_lens=(96 192 336 720)

# SWN Configuration

wavelets=(sym8 sym8 sym8 sym8)
levels=(2 2 2 2)  # Level 3 decomposition is standard for these lengths
d_models=(128 128 128 128) # Hidden size of the spectral MLP
dropouts=(0.2 0.2 0.2 0.2)
decay=(1.0114054433710276e-05 1.0114054433710276e-05 1.0114054433710276e-05 1.0114054433710276e-05)
# Training Config
learning_rates=(0.0021 0.0021 0.0021 0.0021)
batches=(128 128 128 128)
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
