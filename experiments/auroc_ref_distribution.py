import argparse
import os

import json

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType


device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--tokenizer_name", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--dataset_split", type=str, default="test")
parser.add_argument("--dataset_num_skip", type=int, default=0)
parser.add_argument("--data_field", type=str, default="text")
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--num_tokens", type=int, default=200)
parser.add_argument("--streaming", action="store_true", default=False)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--kgw_device", type=str, default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--watermark_configs_file", type=str, required=True)

args = parser.parse_args()

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise Exception(f"Output file {args.output_file} already exists and overwrite_output_file is False")

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split, streaming=args.streaming)

max_length = args.num_tokens
min_length = args.num_tokens

if args.dataset_num_skip > 0:
    dataset = dataset.skip(args.dataset_num_skip)

texts = []
for d in dataset:
    if len(texts) >= args.num_samples:
        break
    tokens = tokenizer(d[args.data_field], truncation=True, max_length=max_length)["input_ids"]
    if len(tokens) >= min_length:
        t = tokenizer.decode(tokens, skip_special_tokens=True)
        texts.append(t)

data = {}

with open(args.watermark_configs_file, "r") as f:
    watermark_configs_list = json.load(f)

for wc in tqdm(watermark_configs_list):
    if wc["type"] == WatermarkType.AAR:
        detector = AarWatermarkDetector(
            k=wc["k"],
            seed=wc["seed"],
            tokenizer=tokenizer,
        )
        watermark_name = f"aar-k{wc['k']}"
    elif wc["type"] == WatermarkType.KGW:
        detector = WatermarkDetector(
            device=wc.get("kgw_device", args.kgw_device),
            tokenizer=tokenizer,
            vocab=tokenizer.get_vocab().values(),
            gamma=wc["gamma"],
            seeding_scheme=wc["seeding_scheme"],
            normalizers=[],
        )
        watermark_name = f"kgw-{wc['seeding_scheme']}-gamma{wc['gamma']}"
    scores = []
    for s in tqdm(texts):
        score = detector.detect(s)
        if wc["type"] == WatermarkType.KGW:
            score = score['p_value']
        scores.append(score)
    data[watermark_name] = {}
    data[watermark_name]["p_values"] = scores
    data[watermark_name]["median_p_value"] = np.median(scores)
    data[watermark_name]["watermark_config"] = wc
    print(f"{watermark_name}\nMedian p-value: {np.median(scores)}")
    del detector

output_dict = {
    "data": data,
}

output_dict.update(vars(args))

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

with open(args.output_file, "w") as f:
    print(f"Writing output to {args.output_file}")
    json.dump(output_dict, f, indent=4)
