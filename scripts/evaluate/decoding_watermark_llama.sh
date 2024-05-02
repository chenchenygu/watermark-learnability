#!/bin/bash
dataset=$1
output_file=$2
llama=$3
ppl_model=$4

num_tokens=200
prompt_length=50
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

python experiments/generate_samples_decoding_watermark.py \
    --model_names "${llama}" \
    ${dataset_args} \
    --streaming \
    --fp16 \
    --output_file "${output_file}" \
    --num_samples ${num_samples} \
    --min_new_tokens ${num_tokens} \
    --max_new_tokens ${num_tokens} \
    --prompt_length ${prompt_length} \
    --watermark_configs_file experiments/watermark-configs/watermark_configs_list.json \
    --batch_size 64 \
    --seed 42

python experiments/compute_metrics.py \
    --input_file "${output_file}"  \
    --output_file "${output_file}" \
    --overwrite_output_file \
    --tokenizer_name "${llama}" \
    --watermark_tokenizer_name "${llama}" \
    --truncate \
    --num_tokens ${num_tokens} \
    --ppl_model_name "${ppl_model}" \
    --fp16 \
    --batch_size 16 \
    --metrics p_value rep ppl

# KTH watermark detection takes a while (several hours) and only requires CPU,
# you can comment this out and run separately if desired
python watermarks/kth/compute_kth_scores.py \
    --tokenizer_name "${llama}" \
    --input_file "${output_file}" \
    --output_file "${output_file}" \
    --num_samples ${num_samples} \
    --num_tokens ${num_tokens} \
    --gamma 0.0 \
    --ref_dist_file "data/${dataset}/kth_ref_distribution_llama_${dataset}.json" \

python experiments/compute_auroc.py \
    --input_file "${output_file}" \
    --output_file "${output_file}" \
    --overwrite_output_file \
    --auroc_ref_dist_file "data/${dataset}/auroc_ref_distribution_llama_${dataset}.json" \
    --kth_ref_dist_file "data/${dataset}/kth_ref_distribution_llama_${dataset}.json"
