#!/bin/bash
dataset=$1
llama=${2:-"meta-llama/Llama-2-7b-hf"}
num_tokens=200
num_samples=5000

if [ "$dataset" = "c4" ]; then
    dataset_args="--dataset_name allenai/c4 \
    --dataset_config_name realnewslike \
    --dataset_split validation \
    --data_field text"
elif [ "$dataset" = "wikipedia" ]; then
    dataset_args="--dataset_name wikipedia \
    --dataset_config_name 20220301.en \
    --dataset_split train \
    --data_field text"
elif [ "$dataset" = "arxiv" ]; then
    dataset_args="--dataset_name scientific_papers \
    --dataset_config_name arxiv \
    --dataset_split test \
    --data_field article"
else
    echo "Unsupported dataset ${dataset}."
    exit 1
fi

python experiments/auroc_ref_distribution.py \
    --tokenizer_name "${llama}" \
    ${dataset_args} \
    --streaming \
    --num_tokens ${num_tokens} \
    --num_samples ${num_samples} \
    --watermark_configs_file "experiments/watermark-configs/auroc_watermark_configs.json" \
    --output_file "data/${dataset}/auroc_ref_distribution_llama_${dataset}.json"
