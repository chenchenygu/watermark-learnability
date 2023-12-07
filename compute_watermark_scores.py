import argparse
import os
import json

import numpy as np
import mauve
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from kgw_watermarking.watermark_reliability_release.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from kth_watermark import KTHWatermark
from aaronson_watermark import AaronsonWatermark, AaronsonWatermarkDetector

DEFAULT_SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--tokenizer_name", type=str, required=True)
parser.add_argument("--watermark_tokenizer_name", type=str, default=None)
parser.add_argument("--num_tokens", type=int, default=200)
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--text_field", type=str, default="full_model_text")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--kgw_device", type=str, default="cpu", choices=["cpu", "cuda"])

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

with open(args.input_file, "r") as f:
    data = json.load(f)

samples_dict = data["samples"]
#prompt_length = data["prompt_length"]

if args.watermark_tokenizer_name is None:
    args.watermark_tokenizer_name = args.tokenizer_name
tokenizer = AutoTokenizer.from_pretrained(args.watermark_tokenizer_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


compute_metrics_args_dict = {}
compute_metrics_args_dict.update(vars(args))
data["compute_metrics_args_dict"] = compute_metrics_args_dict

def save_data():
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w") as f:
        print(f"Writing output to {args.output_file}")
        json.dump(data, f, indent=4)



# compute watermark p-values
for model_name, sd in tqdm(samples_dict.items()):
    if 'watermark_config' in samples_dict[model_name]:
        watermark_config = samples_dict[model_name]['watermark_config']
        if isinstance(watermark_config, list):
            watermark_config = watermark_config[0]
    else:
        #print(f"Skipping {model_name}, no watermark config")
        #continue
        print(f"{model_name}, no watermark config, parsing string")
        watermark_config = {}
    if 'aaronson' in model_name or "k" in watermark_config:
        if not watermark_config:
            aar_s = "aaronson_k"
            k = int(model_name[model_name.find(aar_s) + len(aar_s)])
            seed = DEFAULT_SEED
            print(f"{k=}, {seed=}")
            detector = AaronsonWatermarkDetector(
                k=k,
                seed=seed,
                tokenizer=tokenizer,
            )
        else:
            detector = AaronsonWatermarkDetector(
                k=watermark_config["k"],
                seed=watermark_config.get("seed", DEFAULT_SEED),
                tokenizer=tokenizer,
            )
    elif 'kth' in model_name:
        print(f"Skipping {model_name}, KTH watermark")
        continue
        # detector = KTHWatermark(
        #     vocab_size=watermark_config['vocab_size'],
        #     key_len=watermark_config['key_len'],
        #     seed=watermark_config['seed'],
        #     store_uniform=True,
        #     device="cpu",
        # )
    elif 'kgw' in model_name or "gamma" in watermark_config:
        print(f"gamma = {watermark_config.get('gamma', 0.25)}") 
        detector = WatermarkDetector(
            device=args.kgw_device,
            tokenizer=tokenizer,
            vocab=tokenizer.get_vocab().values(),
            gamma=watermark_config.get('gamma', 0.25),
            seeding_scheme=watermark_config.get('seeding_scheme', "simple_1"),
            normalizers=[],
        )
    else:
        print(f"Skipping {model_name}, didn't match if statements")
        continue
    
    # if 'kth' in model_name:
    #     samples = data['samples'][model_name]['full_model_text']
    # else:
    samples = samples_dict[model_name]['model_text']
    scores = []

    for s in tqdm(samples):
        tokens = tokenizer(
            s,
            add_special_tokens=False,
            truncation=True,
            max_length=args.num_tokens,
        )["input_ids"]
        s = tokenizer.decode(tokens, skip_special_tokens=True)
        score = detector.detect(s)
        if 'kgw' in model_name or "gamma" in watermark_config:
            score = score['p_value']
        scores.append(score)
    sd["p_values"] = scores
    sd["median_p_value"] = np.median(scores)
    print(f"Model name: {model_name}\nMedian p-value: {np.median(scores)}")
    del detector

save_data()