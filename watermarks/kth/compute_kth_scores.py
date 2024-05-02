import argparse
import os
import json

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from watermarks.kth.detect import detect
from watermarks.watermark_types import WatermarkType

parser = argparse.ArgumentParser()

parser.add_argument("--tokenizer_name", type=str, required=True)
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--text_field", type=str, default="model_text")
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--gamma", type=float, default=0.0)
parser.add_argument("--num_tokens", type=int, default=200)
parser.add_argument("--ref_dist_file", type=str, default=None)

args = parser.parse_args()

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

with open(args.input_file, "r") as f:
    data = json.load(f)

samples_dict = data["samples"]

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

vocab_size = len(tokenizer)

if args.ref_dist_file is not None:
    with open(args.ref_dist_file) as f:
        ref_dist_data = json.load(f)

    ref_dist = ref_dist_data["test_stat_ref_dist"]
    ref_dist = np.array(ref_dist)
    for i in range(len(ref_dist)):
        if ref_dist[i] == float('-inf'):
            ref_dist[i] = np.median(ref_dist)
    assert min(ref_dist) != float('-inf')
else:
    ref_dist = None

compute_kth_scores_args_dict = {}
compute_kth_scores_args_dict.update(vars(args))
data["compute_kth_scores_args_dict"] = compute_kth_scores_args_dict

def save_data():
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        print(f"Writing output to {args.output_file}")
        json.dump(data, f, indent=4)

# compute watermark p-values
for model_name, sd in tqdm(samples_dict.items()):
    if "watermark_config" in samples_dict[model_name]:
        watermark_config = samples_dict[model_name]["watermark_config"]
    else:
        watermark_config = {}

    if watermark_config.get("type") != WatermarkType.KTH and WatermarkType.KTH not in watermark_config:
        continue

    seed = watermark_config["seed"]
    key_len = watermark_config["key_len"]
    vocab_size = watermark_config.get("vocab_size", vocab_size)

    generator = torch.Generator()  # generator is always cpu for reproducibility
    generator.manual_seed(seed)

    xi = torch.rand((key_len, vocab_size), generator=generator, dtype=torch.float32)
    xi = xi.numpy()

    test_stats = []
    samples = sd[args.text_field]

    for text in tqdm(samples):
        if len(test_stats) >= args.num_samples:
            break
        tokens = tokenizer.encode(text, return_tensors='np', add_special_tokens=False)[0]
        if len(tokens) < args.num_tokens:
            continue
        tokens = tokens[:args.num_tokens]
        null_result = detect(tokens, len(xi), len(tokens), xi, gamma=args.gamma)
        test_stats.append(null_result)
    
    sd["kth_test_stats"] = test_stats
    print(f"{model_name} median test stat: {np.median(test_stats)}")
    sd["median_kth_test_stat"] = np.median(test_stats)
    print(f"{len(test_stats)} samples")
    if ref_dist is not None:
        p_values = []
        for ts in test_stats:
            p_val = (1 + np.sum(ref_dist < ts)) / (len(ref_dist) + 1)
            p_values.append(p_val)
        assert len(p_values) == len(test_stats)
        sd["p_values"] = p_values
        sd["median_p_value"] = np.median(p_values)
        print(f"{model_name} median p value: {np.median(p_values)}")
    del xi
    save_data()


os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

with open(args.output_file, "w") as f:
    print(f"Writing output to {args.output_file}")
    json.dump(data, f, indent=4)
