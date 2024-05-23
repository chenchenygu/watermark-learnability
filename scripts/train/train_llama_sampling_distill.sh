#!/bin/bash
watermark=$1
out_dir=$2
port=$3
llama=${4:-"meta-llama/Llama-2-7b-hf"}
dataset_origin=${5:-"hf"}

watermark_config_file="experiments/watermark-configs/${watermark}-config.json"
model_name="llama-2-7b-sampling-watermark-distill-${watermark}"

if [ "$dataset_origin" = "hf" ]; then
    dataset_args="--dataset_name cygu/sampling-distill-train-data-${watermark}"
elif [ "$dataset_origin" = "local" ]; then
    dataset_args="--train_file data/sampling-distill-train-data/sampling-distill-train-data-${watermark}.json"
else
    echo "dataset_origin must be either \"hf\" or \"local\". Received ${dataset_origin}."
    exit 1
fi

if [[ "$watermark" == kth* ]]; then
    group_texts="False"
else
    group_texts="True"
fi

torchrun --nproc_per_node=4 --master_port=${port} train_sampling_distill.py \
    --model_name_or_path "${llama}" \
    ${dataset_args} \
    --watermark_config_file "${watermark_config_file}" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --logging_steps 1 \
    --output_dir "${out_dir}${model_name}" \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --block_size 256 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --num_train_epochs 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    --group_texts ${group_texts} \
    --tf32 True \
    --bf16 True
