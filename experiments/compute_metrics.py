import argparse
import os
import json

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from watermarks.kgw.watermark_processor import WatermarkDetector
from watermarks.aar.aar_watermark import AarWatermarkDetector
from watermarks.watermark_types import WatermarkType

DEFAULT_SEED = 42
METRICS = ["p_value", "rep", "ppl"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--watermark_tokenizer_name", type=str, default=None)
    parser.add_argument("--truncate", action="store_true", default=False)
    parser.add_argument("--num_tokens", type=int, default=200)
    parser.add_argument("--ppl_model_name", type=str)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for compting perplexity.")
    parser.add_argument("--overwrite_output_file", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--kgw_device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--metrics", type=str, nargs="+", default=METRICS, choices=METRICS)

    args = parser.parse_args()
    return args


def compute_p_values(samples_dict, tokenizer, kgw_device, truncate=False, num_tokens=200):
    """Compute watermark detection p-values."""
    for model_name, sd in tqdm(samples_dict.items()):
        if "watermark_config" in samples_dict[model_name]:
            watermark_config = samples_dict[model_name]["watermark_config"]
            if isinstance(watermark_config, list):
                watermark_config = watermark_config[0]
        else:
            print(f"Skipping {model_name}, no watermark config")
            continue

        if "type" not in watermark_config:
            print(f"Skipping {model_name}, watermark type not specified in config")
            continue

        if watermark_config["type"] == WatermarkType.AAR:
            watermark_type = WatermarkType.AAR
            detector = AarWatermarkDetector(
                k=watermark_config["k"],
                seed=watermark_config.get("seed", DEFAULT_SEED),
                tokenizer=tokenizer,
            )
        elif watermark_config["type"] == WatermarkType.KTH:
            # KTH detection in watermarks/kth/compute_kth_scores.py, takes long time, CPU bound
            print(f"Skipping {model_name}, KTH watermark")
            continue
        elif watermark_config["type"] == WatermarkType.KGW:
            watermark_type = WatermarkType.KGW
            detector = WatermarkDetector(
                device=watermark_config.get("kgw_device", kgw_device),
                tokenizer=tokenizer,
                vocab=tokenizer.get_vocab().values(),
                gamma=watermark_config["gamma"],
                seeding_scheme=watermark_config["seeding_scheme"],
                normalizers=[],
            )
        else:
            print(f"Skipping {model_name}, could not determine watermark type")
            continue
        
        samples = samples_dict[model_name]["model_text"]
        scores = []

        for s in tqdm(samples):
            if truncate:
                tokens = tokenizer(
                    s,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=num_tokens,
                )["input_ids"]
                s = tokenizer.decode(tokens, skip_special_tokens=True)
            score = detector.detect(s)
            if watermark_type == WatermarkType.KGW:
                score = score["p_value"]
            scores.append(score)
        sd["p_values"] = scores
        sd["median_p_value"] = np.median(scores)
        print(f"Model name: {model_name}\nMedian p-value: {np.median(scores)}")
        del detector


def compute_seq_rep_n(samples, tokenizer, n=3):
    """compute seq-rep-n metric"""
    n_gram_reps = []
    
    for s in samples:
        n_grams = []
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))
                    
        rep = 1 - len(set(n_grams)) / len(n_grams)
        n_gram_reps.append(rep)
            
    median_rep = np.median(n_gram_reps)
    mean_rep = np.mean(n_gram_reps)
    return {
        f"median_seq_rep_{n}": median_rep,
        f"mean_seq_rep_{n}": mean_rep,
        f"list_seq_rep_{n}": n_gram_reps,
    }


def compute_total_rep_n(samples, tokenizer, n=3):
    """compute total-rep-n metric"""
    n_grams = []
    
    for s in samples:
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))

    total_rep = 1 - len(set(n_grams)) / len(n_grams)        

    return {f"total_rep_{n}": total_rep}


def compute_repetition(samples_dict, tokenizer):
    """Compute repetition metrics."""
    for model_name, sd in tqdm(samples_dict.items()):
        samples = samples_dict[model_name]["model_text"]
        sd.update(compute_seq_rep_n(samples, tokenizer, n=3))
        sd.update(compute_total_rep_n(samples, tokenizer, n=3))
        print(f"Model name: {model_name}\nMedian seq rep 3: {sd['median_seq_rep_3']}\nTotal rep 3: {sd['total_rep_3']}")


def compute_ppl(samples_dict, ppl_model_name, batch_size, fp16=True):
    """Compute perplexities under `ppl_model_name`."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(ppl_model_name).to(device)
    if fp16:
        model = model.half()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ppl_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, sd in tqdm(samples_dict.items()):
        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        samples = sd["full_model_text"]
        prompts = sd["prompt_text"]

        for i in tqdm(range(0, len(samples), batch_size)):
            s = samples[i:i + batch_size]
            encodings = tokenizer(
                s,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)

            encoded_batch = encodings["input_ids"]
            attn_mask = encodings["attention_mask"]

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            prompt_text = prompts[i:i + batch_size]
            prompt_encodings = tokenizer(
                prompt_text,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)
            prompt_attn_mask = prompt_encodings["attention_mask"]

            # match shape of prompt_attn_mask and attn_mask by padding with 0
            padding = torch.zeros(
                (attn_mask.shape[0], attn_mask.shape[1] - prompt_attn_mask.shape[1]),
            ).to(device)
            padded_prompt_attn_mask = torch.cat([prompt_attn_mask, padding], dim=1)
            prompt_mask = (padded_prompt_attn_mask == 1)
            
            # don't score prompt tokens
            attn_mask[prompt_mask] = 0

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        mean_perplexity = np.mean(ppls)
        median_perplexity = np.median(ppls)
        sd["mean_perplexity"] = mean_perplexity
        sd["median_perplexity"] = median_perplexity
        sd["perplexities"] = ppls
        print(f"model name: {name}")
        print(f"mean perplexity: {mean_perplexity}")
        print(f"median perplexity: {median_perplexity}")


def save_data(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        print(f"Writing output to {output_file}")
        json.dump(data, f, indent=4)


def main():
    args = parse_args()
    if os.path.exists(args.output_file) and not args.overwrite_output_file:
        raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

    with open(args.input_file, "r") as f:
        data = json.load(f)

    compute_metrics_args_dict = {}
    compute_metrics_args_dict.update(vars(args))
    data["compute_metrics_args_dict"] = compute_metrics_args_dict

    samples_dict = data["samples"]

    if args.watermark_tokenizer_name is None:
        args.watermark_tokenizer_name = args.tokenizer_name
    watermark_tokenizer = AutoTokenizer.from_pretrained(args.watermark_tokenizer_name)

    if watermark_tokenizer.pad_token is None:
        watermark_tokenizer.pad_token = watermark_tokenizer.eos_token

    if "p_value" in args.metrics:
        compute_p_values(samples_dict, watermark_tokenizer, args.kgw_device, args.truncate, args.num_tokens)
        save_data(data, args.output_file)

    # switch to model generated tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "rep" in args.metrics:
        compute_repetition(samples_dict, tokenizer)
        save_data(data, args.output_file)

    if "ppl" in args.metrics:
        compute_ppl(samples_dict, args.ppl_model_name, args.batch_size, args.fp16)
        save_data(data, args.output_file)
    

if __name__ == "__main__":
    main()
