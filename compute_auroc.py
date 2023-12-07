import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser()

parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--ref_dist_file", type=str, required=True)
parser.add_argument("--kth_ref_dist_file", type=str, required=True)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)

args = parser.parse_args()

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise ValueError(f"Output file {args.output_file} already exists and overwrite_output_file is False")

with open(args.input_file, "r") as f:
    data = json.load(f)
    samples_dict = data["samples"]

with open(args.ref_dist_file, "r") as f:
    ref_dist_data = json.load(f)
    ref_dist = ref_dist_data["data"]

with open(args.kth_ref_dist_file, "r") as f:
    kth_ref_dist_data = json.load(f)
    kth_ref_dist = kth_ref_dist_data['test_stat_ref_dist']
    kth_ref_dist = np.array(kth_ref_dist)
    for i in range(len(kth_ref_dist)):
        if kth_ref_dist[i] == float('-inf'):
            kth_ref_dist[i] = np.median(kth_ref_dist)
    assert min(kth_ref_dist) != float('-inf')

for model_name, sd in samples_dict.items():
    print(model_name)
    watermark_scores = None
    if "kth_test_stats" not in sd and "p_values" not in sd:
        print(f"Skipping {model_name}, p_values/test-stats not computed")
        continue
    if "kth" in model_name:
        print("kth")
        watermark_scores = sd["kth_test_stats"]
        null_scores = kth_ref_dist
    else:
        for name, wd in ref_dist.items():
            if name in model_name or name.replace("_", "-") in model_name:
                print(name)
                watermark_scores = sd["p_values"]
                null_scores = wd["p_values"]
                break
    if watermark_scores is None:
        print(f"Skipping {model_name}")
        continue
    null_scores = null_scores[:len(watermark_scores)]
    y_true = np.concatenate([np.zeros_like(watermark_scores), np.ones_like(null_scores)])
    y_score = np.concatenate([watermark_scores, null_scores])
    auroc = roc_auc_score(y_true, y_score)
    print(auroc)
    sd["auroc"] = auroc

os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

with open(args.output_file, "w") as f:
    print(f"Writing output to {args.output_file}")
    json.dump(data, f, indent=4)
