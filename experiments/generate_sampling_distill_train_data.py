import argparse
import os
import random

import json
from typing import Dict
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, set_seed

from watermarks.aar.aar_watermark import AarWatermark
from watermarks.kgw.watermark_processor import WatermarkLogitsProcessor
from watermarks.kth.kth_watermark import KTHWatermark
from watermarks.watermark_types import WatermarkType


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--tokenizer_name", type=str, default=None)
parser.add_argument("--dataset_config_name", type=str, default=None)
parser.add_argument("--dataset_split", type=str, default="train")
parser.add_argument("--dataset_num_skip", type=int, default=0)
parser.add_argument("--data_field", type=str, default="text")
parser.add_argument("--num_samples", type=int, required=True)
parser.add_argument("--min_new_tokens", type=int, default=256)
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--prompt_length", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--streaming", action="store_true", default=True)
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--output_train_file", type=str, required=True)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--watermark_config_file", type=str, required=True)
parser.add_argument("--save_interval", type=int, default=64000)
parser.add_argument("--dataloader_batch_size", type=int, default=10000)

args = parser.parse_args()


def get_prompts(args, additional_num_skip: int = 0) -> Dict:
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)  

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split, streaming=args.streaming)

    def encode(examples):
        prompt = tokenizer(
            examples[args.data_field], truncation=True, padding=True, max_length=args.prompt_length, return_tensors="pt",
        ).to(device)
        examples["prompt_text"] = tokenizer.batch_decode(prompt["input_ids"], skip_special_tokens=True)
        examples["input_ids"] = prompt["input_ids"]
        examples["attention_mask"] = prompt["attention_mask"]
        return examples

    dataset = dataset.skip(args.dataset_num_skip)
    if additional_num_skip > 0:
        dataset = dataset.skip(additional_num_skip)
    dataset = dataset.map(encode, batched=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.dataloader_batch_size)

    input_ids_list = []
    attention_mask_list = []
    prompt_text = []
    for batch in tqdm(dataloader):
        if len(prompt_text) >= args.num_samples - additional_num_skip:
            break
        input_ids_list.extend(torch.split(batch["input_ids"], 1, dim=0))
        attention_mask_list.extend(torch.split(batch["attention_mask"], 1, dim=0))
        prompt_text.extend(batch["prompt_text"])
    batched_prompts = []
    for i in range(0, len(input_ids_list), args.batch_size):
        batch = {
            "input_ids": torch.cat(input_ids_list[i:i+args.batch_size], dim=0),
            "attention_mask": torch.cat(attention_mask_list[i:i+args.batch_size], dim=0),
        }
        batched_prompts.append(batch)
    return {
        "prompts": batched_prompts,
        "prompt_text": prompt_text,
    }


def generate_samples(model, tokenizer, args, prompts, watermark, do_sample=True) -> Dict:
    model_text = []

    for batch in tqdm(prompts):
        if len(model_text) >= args.num_samples:
            break
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                do_sample=do_sample,
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                logits_processor=LogitsProcessorList([watermark]),
            )

            n_input_tokens = batch["input_ids"].shape[1]
            model_text.extend(tokenizer.batch_decode(outputs[:, n_input_tokens:], skip_special_tokens=True))

    del model
    torch.cuda.empty_cache()
    
    # model_text discards the prompt, full_model_text contains the prompt
    return {"model_text": model_text}


if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

if os.path.exists(args.output_train_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_train_file} already exists and overwrite_output_file is False")

if args.input_file and os.path.exists(args.input_file):
    with open(args.input_file, "r") as f:
        input_dict = json.load(f)
    samples_dict = input_dict["samples"]
else:
    samples_dict = {}

if samples_dict:
    temp_key = list(samples_dict.keys())[0]
    num_samples_so_far = len(samples_dict[temp_key]["model_text"])
    prompts_dict = get_prompts(args, additional_num_skip=num_samples_so_far)
else:
    prompts_dict = get_prompts(args)

prompts = prompts_dict["prompts"]
prompt_text = prompts_dict["prompt_text"]

if samples_dict:
    temp_key = list(samples_dict.keys())[0]
    prompt_text = samples_dict[temp_key]["prompt_text"] + prompt_text

with open(args.watermark_config_file, "r") as f:
    watermark_config = json.load(f)


output_dict = {
    "samples": samples_dict,
}
output_dict.update(vars(args))

model_name = args.model_name

set_seed(args.seed)

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

if watermark_config["type"] == WatermarkType.AAR:
    watermark = AarWatermark(
        vocab_size=len(tokenizer),
        k=watermark_config["k"],
        seed=watermark_config["seed"],
        device=device,
    )
    do_sample = False
elif watermark_config["type"] == WatermarkType.KGW:
    watermark = WatermarkLogitsProcessor(
        vocab=tokenizer.get_vocab().values(),
        gamma=watermark_config["gamma"],
        delta=watermark_config["delta"],
        seeding_scheme=watermark_config["seeding_scheme"],
        device=device,
    )
    do_sample = True
elif watermark_config["type"] == WatermarkType.KTH:
    watermark = KTHWatermark(
        vocab_size=len(tokenizer),
        key_len=watermark_config["key_len"],
        seed=watermark_config["seed"],
        num_shifts=watermark_config["num_shifts"],
        device=device,
    )
    do_sample = False
else:
    raise ValueError(f"Invalid watermark type {watermark_config['type']}")

simplified_model_name = [s for s in model_name.split("/") if s][-1]
simplified_model_name = watermark_config["type"] + "_" + simplified_model_name
print(f"Generating samples for model {simplified_model_name}")
if simplified_model_name in samples_dict:
    print(f"Loaded saved samples for {simplified_model_name}")

model_text = samples_dict.get(simplified_model_name, {}).get("model_text", [])

samples = {}
samples["model_text"] = model_text

watermark_config_vars = {}
try:
    watermark_config_vars = {}
    for k, v in vars(watermark).items():
        if isinstance(v, (str, int, float, bool, list)):
            watermark_config_vars[k] = v
except Exception as e:
    print(f"Error loading watermark config for model {model_name}: {e}")
if watermark_config_vars:
    samples["watermark_config_vars"] = watermark_config_vars
samples["prompt_text"] = prompt_text
samples["model_name"] = simplified_model_name
samples_dict[simplified_model_name] = samples
samples["watermark_config"] = watermark_config

prev_save = 0

for batch in tqdm(prompts):
    if len(model_text) >= args.num_samples:
        break
    if len(model_text) >= prev_save + args.save_interval:
        prev_save = len(model_text)
        save_filename = f"{args.output_file.rsplit('.', maxsplit=1)[0]}_save-{prev_save}.json"
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)

        with open(save_filename, "w") as f:
            print(f"Writing output to {save_filename}, save interval={prev_save}")
            json.dump(output_dict, f, indent=4)

    if watermark_config["type"] == WatermarkType.KTH and watermark.num_shifts > 1:
        watermark.cur_shift = random.choice(watermark.possible_shifts)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=do_sample,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            logits_processor=LogitsProcessorList([watermark]),
            pad_token_id=tokenizer.eos_token_id,
        )

        n_input_tokens = batch["input_ids"].shape[1]
        model_text.extend(tokenizer.batch_decode(outputs[:, n_input_tokens:], skip_special_tokens=True))

del model
torch.cuda.empty_cache()

del watermark

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

with open(args.output_file, "w") as f:
    print(f"Writing output to {args.output_file}")
    json.dump(output_dict, f, indent=4)

train_file_data = []
for s in model_text:
    train_file_data.append(json.dumps({"text": s}))

with open(args.output_train_file, "w") as f:
    print(f"Writing output to {args.output_train_file}")
    f.write("\n".join(train_file_data))
