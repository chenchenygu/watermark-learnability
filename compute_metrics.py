import argparse
import os
import json

import numpy as np
import mauve
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from kgw_watermarking.watermark_reliability_release.watermark_processor import WatermarkDetector
from aar_watermark import AarWatermarkDetector

DEFAULT_SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--tokenizer_name", type=str, required=True)
parser.add_argument("--watermark_tokenizer_name", type=str, default=None)
parser.add_argument("--truncate", action="store_true", default=False)
parser.add_argument("--num_tokens", type=int, default=200)
parser.add_argument("--lm_score_model_name", type=str, required=True)
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--text_field", type=str, default="full_model_text")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--kgw_device", type=str, default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--mauve_max_length", type=int, default=200)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

with open(args.input_file, "r") as f:
    data = json.load(f)

samples_dict = data["samples"]
prompt_length = data["prompt_length"]

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
    if 'aar' in model_name or "k" in watermark_config:
        if not watermark_config:
            aar_s = "aar-k"
            k = int(model_name[model_name.find(aar_s) + len(aar_s)])
            seed = DEFAULT_SEED
            print(f"{k=}, {seed=}")
            detector = AarWatermarkDetector(
                k=k,
                seed=seed,
                tokenizer=tokenizer,
            )
        else:
            detector = AarWatermarkDetector(
                k=watermark_config["k"],
                seed=watermark_config.get("seed", DEFAULT_SEED),
                tokenizer=tokenizer,
            )
    elif 'kth' in model_name:
        # KTH detection in kth_watermarking/compute_kth_scores.py, takes long time
        print(f"Skipping {model_name}, KTH watermark")
        continue
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
    
    samples = samples_dict[model_name]['model_text']
    scores = []

    for s in tqdm(samples):
        if args.truncate:
            tokens = tokenizer(
                s,
                add_special_tokens=False,
                truncation=True,
                max_length=args.num_tokens,
            )["input_ids"]
            s = tokenizer.decode(tokens, skip_special_tokens=True)
        score = detector.detect(s)
        if 'kgw' in model_name:
            score = score['p_value']
        scores.append(score)
    sd["p_values"] = scores
    sd["median_p_value"] = np.median(scores)
    print(f"Model name: {model_name}\nMedian p-value: {np.median(scores)}")
    del detector

save_data()

# switch to model generated tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

def compute_seq_rep_n(samples, n=3):
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

def compute_total_rep_n(samples, n=3):
    """compute total-rep-n metric"""
    n_grams = []
    
    for s in samples:
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))

    total_rep = 1 - len(set(n_grams)) / len(n_grams)        

    return {f"total_rep_{n}": total_rep}


# compute repetition metrics
for model_name, sd in tqdm(samples_dict.items()):
    samples = samples_dict[model_name]['model_text']
    sd.update(compute_seq_rep_n(samples, n=3))
    sd.update(compute_total_rep_n(samples, n=3))
    print(f"Model name: {model_name}\nMedian seq rep 3: {sd['median_seq_rep_3']}\nTotal rep 3: {sd['total_rep_3']}")

save_data()

# compute MAUVE score
m = list(samples_dict.keys())[0]
full_human_text = samples_dict[m]['full_human_text']

full_p_text = []
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')

for s in full_human_text:
    tokens = gpt2_tokenizer(s, truncation=True, max_length=args.mauve_max_length, add_special_tokens=False)["input_ids"]
    if len(tokens) >= args.mauve_max_length:
        full_p_text.append(gpt2_tokenizer.decode(tokens, skip_special_tokens=True))

for model_name, sd in tqdm(samples_dict.items()):
    q_text = []
    for s in samples_dict[model_name]['full_model_text']:
        tokens = gpt2_tokenizer(s, truncation=True, max_length=args.mauve_max_length, add_special_tokens=False)["input_ids"]
        if len(tokens) >= args.mauve_max_length:
            q_text.append(s)
            
    p_text = full_p_text[:min(len(full_p_text), len(q_text))]
    q_text = q_text[:min(len(full_p_text), len(q_text))]

    out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=args.mauve_max_length, verbose=False)
    print(f"{model_name} MAUVE: {out.mauve}")
    sd["mauve"] = out.mauve


save_data()


model = AutoModelForCausalLM.from_pretrained(args.lm_score_model_name).to(device)
if args.fp16:
    model = model.half()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.lm_score_model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# compute LM score
for name, sd in tqdm(samples_dict.items()):
    if "perplexities" in sd:
        continue

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    samples = sd[args.text_field]

    for i in tqdm(range(0, len(samples), args.batch_size)):
        s = samples[i:i + args.batch_size]

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

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        # don't score prompt tokens
        shift_logits = shift_logits[:, prompt_length:, :].contiguous()
        shift_labels = shift_labels[:, prompt_length:].contiguous()
        shift_attention_mask_batch = shift_attention_mask_batch[:, prompt_length:].contiguous()

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


save_data()
