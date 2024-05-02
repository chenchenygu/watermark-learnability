import argparse
import os
import json

import numpy as np
from transformers import AutoTokenizer

DEFAULT_SEED = 42
P_EDIT_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--overwrite_output_file", action="store_true", default=False)
    parser.add_argument("--p_edits", type=float, nargs="+", default=P_EDIT_LIST)
    args = parser.parse_args()
    return args


def random_edit(
        s: str, tokenizer, p_edit: float, rng: np.random.Generator, min_random_token_id: int = 10
    ) -> str:
    vocab_size = len(tokenizer)
    tokens = np.array(tokenizer(s, add_special_tokens=False).input_ids)
    n_tokens = len(tokens)
    n_edits = round(n_tokens * p_edit)
    
    # randomly choose which tokens to keep
    orig_mask = np.full(n_tokens, True)
    orig_mask[:n_edits] = False
    rng.shuffle(orig_mask)

    # min_random_token_id ensure that special tokens are not inserted, e.g. EOS token
    new_tokens = rng.integers(min_random_token_id, vocab_size - min_random_token_id, size=n_tokens)

    # insert random tokens at random positions
    new_mask = np.full(n_tokens, True)
    new_mask[:n_edits] = False
    rng.shuffle(new_mask)
    new_tokens[new_mask] = tokens[orig_mask]
            
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def random_edits_all(samples_dict, tokenizer_name, p_edit_list, seed=DEFAULT_SEED):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    new_samples_dict = {}
    for model_name, sd in samples_dict.items():
        original_samples = sd["model_text"]
        for p_edit in p_edit_list:
            new_model_name = f"{p_edit}edit-{model_name}"
            new_sd = sd.copy()
            new_samples = []
            rng = np.random.default_rng(seed)
            for s in original_samples:
                new_samples.append(random_edit(s, tokenizer, p_edit, rng))
            if p_edit == 0.0:
                new_samples = original_samples
            new_sd["model_text"] = new_samples
            new_sd["p_edit"] = p_edit
            new_sd["original_model_name"] = model_name
            new_samples_dict[new_model_name] = new_sd
            print(f"{new_model_name}")
    return new_samples_dict


def main():
    args = parse_args()
    if os.path.exists(args.output_file) and not args.overwrite_output_file:
        raise Exception(f"Output file {args.output_file} already exists and overwrite_output_file is False")

    with open(args.input_file, "r") as f:
        data = json.load(f)

    samples_dict = data["samples"]
    new_samples_dict = random_edits_all(samples_dict, args.tokenizer_name, args.p_edits)
    data["samples"] = new_samples_dict

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        print(f"Writing output to {args.output_file}")
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
