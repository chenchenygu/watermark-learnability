#!/bin/bash
dataset=$1
llama=${2:-"meta-llama/Llama-2-7b-hf"}
num_tokens=200
prompt_length=50

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
    --dataset_split train \
    --data_field article"
else
    echo "Unsupported dataset ${dataset}."
    exit 1
fi

python watermarks/kth/kth_ref_distribution.py \
    --tokenizer_name "${llama}" \
    ${dataset_args} \
    --streaming \
    --num_samples 10000 \
    --prompt_length ${prompt_length} \
    --completion_length ${num_tokens} \
    --key_len 256 \
    --seed 42 \
    --gamma 0.0 \
    --output_file "data/${dataset}/kth_ref_distribution_llama_${dataset}.json"
