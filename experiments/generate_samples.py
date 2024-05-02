import argparse
import os

import json
from typing import Dict
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, nargs="+", required=True)
parser.add_argument("--watermark_config_filename", type=str, default="watermark_config.json")
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--tokenizer_name", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--dataset_split", type=str, default="test")
parser.add_argument("--dataset_num_skip", type=int, default=0)
parser.add_argument("--data_field", type=str, default="text")
parser.add_argument("--num_samples", type=int, default=5000)
parser.add_argument("--min_new_tokens", type=int, default=200)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--prompt_length", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--streaming", action="store_true", default=False)
parser.add_argument("--greedy", action="store_true", default=False)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--fp16", action="store_true", default=False)

args = parser.parse_args()

DO_SAMPLE = True
if args.greedy is True:
    DO_SAMPLE = False


def get_prompts(args) -> Dict:
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_names[0])  

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split, streaming=args.streaming)

    max_length = args.prompt_length + args.max_new_tokens
    min_length = args.prompt_length + args.min_new_tokens

    def filter_length(example):
        return len(tokenizer(example[args.data_field], truncation=True, max_length=max_length)["input_ids"]) >= min_length

    def encode(examples):
        trunc_tokens = tokenizer(
            examples[args.data_field],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        examples["text"] = tokenizer.batch_decode(trunc_tokens["input_ids"], skip_special_tokens=True)
        prompt = tokenizer(
            examples["text"], truncation=True, padding=True, max_length=args.prompt_length, return_tensors="pt",
        ).to(device)
        examples["prompt_text"] = tokenizer.batch_decode(prompt["input_ids"], skip_special_tokens=True)
        examples["input_ids"] = prompt["input_ids"]
        examples["attention_mask"] = prompt["attention_mask"]
        examples["text_completion"] = tokenizer.batch_decode(
            trunc_tokens["input_ids"][:, args.prompt_length:], skip_special_tokens=True
        )
        return examples

    dataset = dataset.filter(filter_length)
    if args.dataset_num_skip > 0:
        dataset = dataset.skip(args.dataset_num_skip)
    dataset = dataset.map(encode, batched=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    prompts = []
    human_text = []
    prompt_text = []
    full_human_text = []
    for batch in dataloader:
        if len(human_text) >= args.num_samples:
            break
        if (type(batch["input_ids"]) == list):
            batch["input_ids"] = torch.stack(batch["input_ids"], dim=1).to(device)
        if (type(batch["attention_mask"]) == list):
            batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=1).to(device)
        prompts.append(batch)
        human_text.extend(batch["text_completion"])
        prompt_text.extend(batch["prompt_text"])
        full_human_text.extend(batch["text"])
    human_text = human_text[:args.num_samples]
    prompt_text = prompt_text[:args.num_samples]
    full_human_text = full_human_text[:args.num_samples]
    return {
        "prompts": prompts,
        "human_text": human_text,
        "prompt_text": prompt_text,
        "full_human_text": full_human_text,
    }


def generate_samples(model_name, args, prompts) -> Dict:
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)  
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    if args.fp16:
        model = model.half()
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_text = []
    full_model_text = []

    for batch in tqdm(prompts):
        if len(model_text) >= args.num_samples:
            break
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=DO_SAMPLE,
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.eos_token_id,
            )

            n_input_tokens = batch["input_ids"].shape[1]
            model_text.extend(tokenizer.batch_decode(outputs[:, n_input_tokens:], skip_special_tokens=True))
            full_model_text.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()
    
    # model_text discards the prompt, full_model_text contains the prompt
    model_text = model_text[:args.num_samples]
    full_model_text = full_model_text[:args.num_samples]
    return {"model_text": model_text, "full_model_text": full_model_text}


if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

if os.path.exists(args.output_file):
    with open(args.output_file, "r") as f:
        input_dict = json.load(f)
    for key in input_dict:
        if key in args and "model_name" not in key and "file" not in key:
            setattr(args, key, input_dict[key])
    samples_dict = input_dict["samples"]
else:
    samples_dict = {}

prompts_dict = get_prompts(args)
prompts = prompts_dict["prompts"]
human_text = prompts_dict["human_text"]
prompt_text = prompts_dict["prompt_text"]
full_human_text = prompts_dict["full_human_text"]

for model_name in tqdm(args.model_names):
    set_seed(args.seed)
    simplified_model_name = [s for s in model_name.split("/") if s][-1]
    print(f"Generating samples for model {simplified_model_name}")
    if simplified_model_name in samples_dict:
        print(f"Skipping model {simplified_model_name} because samples already generated")
        continue

    try:
        samples = generate_samples(model_name, args, prompts)
    except Exception as e:
        print(f"Error generating samples for model {model_name}: {e}")
        continue

    samples["human_text"] = human_text
    samples["prompt_text"] = prompt_text
    samples["full_human_text"] = full_human_text
    watermark_config = {}
    try:
        with open(os.path.join(model_name, args.watermark_config_filename), "r") as f:
            watermark_config = json.load(f)
    except Exception as e:
        print(f"Error loading watermark config for model {model_name}: {e}")
    if watermark_config:
        samples["watermark_config"] = watermark_config
    samples["model_name"] = simplified_model_name
    samples_dict[simplified_model_name] = samples

output_dict = {
    "samples": samples_dict,
}
output_dict.update(vars(args))

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

with open(args.output_file, "w") as f:
    print(f"Writing output to {args.output_file}")
    json.dump(output_dict, f, indent=4)
