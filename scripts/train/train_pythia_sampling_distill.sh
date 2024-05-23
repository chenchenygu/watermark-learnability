#!/bin/bash
watermark=$1
out_dir=$2
port=$3
pythia=${4:-"EleutherAI/pythia-1.4b"}
dataset_location=${5:-"hf"}

watermark_config_file="experiments/watermark-configs/${watermark}-config.json"
model_name="pythia-1.4b-sampling-watermark-distill-${watermark}"

if [ "$dataset_location" = "hf" ]; then
    dataset_args="--dataset_name cygu/sampling-distill-train-data-${watermark}"
elif [ "$dataset_location" = "local" ]; then
    dataset_args="--train_file data/sampling-distill-train-data/sampling-distill-train-data-${watermark}.json"
else
    echo "dataset_location must be either \"hf\" or \"local\". Received ${dataset_location}."
    exit 1
fi

if [[ "$watermark" == kth* ]]; then
    group_texts="False"
else
    group_texts="True"
fi

torchrun --nproc_per_node=1 --master_port=${port} train_sampling_distill.py \
    --model_name_or_path "${pythia}" \
    ${dataset_args} \
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
