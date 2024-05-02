#!/bin/bash
watermark=$1
out_dir=$2
port=$3
pythia=${4:-"EleutherAI/pythia-1.4b"}

train_file="data/sampling-distill-train-data/${watermark}_llama_2_7b_owt_len256_640k_train.json"
watermark_config_file="experiments/watermark-configs/${watermark}-config.json"
model_name="pythia-1.4b-sampling-watermark-distill-${watermark}"

if [[ "$watermark" == kth* ]]; then
    group_texts="False"
else
    group_texts="True"
fi

torchrun --nproc_per_node=1 --master_port=${port} train_sampling_distill.py \
    --model_name_or_path "${pythia}" \
    --train_file "${train_file}" \
    --watermark_config_file "${watermark_config_file}" \
    --per_device_train_batch_size 64 \
    --do_train \
    --logging_steps 1 \
    --output_dir "${out_dir}${model_name}" \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --block_size 256 \
    --save_steps 2500 \
    --save_total_limit 1 \
    --num_train_epochs 1 \
    --group_texts ${group_texts} \
    --tf32 True \
    --bf16 True
