import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from watermarks.watermark_types import WatermarkType


parser = argparse.ArgumentParser()

parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--auroc_ref_dist_file", type=str, required=True)
parser.add_argument("--kth_ref_dist_file", type=str, required=True)
parser.add_argument("--overwrite_output_file", action="store_true", default=False)

args = parser.parse_args()

if os.path.exists(args.output_file) and not args.overwrite_output_file:
    raise Exception(f"Output file {args.output_file} already exists and overwrite_output_file is False")

with open(args.input_file, "r") as f:
    data = json.load(f)
    samples_dict = data["samples"]

with open(args.auroc_ref_dist_file, "r") as f:
    ref_dist_data = json.load(f)
    ref_dist = ref_dist_data["data"]

with open(args.kth_ref_dist_file, "r") as f:
    kth_ref_dist_data = json.load(f)
    kth_ref_dist = kth_ref_dist_data["test_stat_ref_dist"]
    kth_ref_dist = np.array(kth_ref_dist)
    for i in range(len(kth_ref_dist)):
        if kth_ref_dist[i] == float('-inf'):
            kth_ref_dist[i] = np.median(kth_ref_dist)
    assert min(kth_ref_dist) != float('-inf')

for model_name, sd in samples_dict.items():
    print(model_name)
    watermark_scores = None
    if "watermark_config" not in sd:
        print(f"Skipping {model_name}, no watermark_config")
        continue
    wc = sd["watermark_config"]
    if "kth_test_stats" not in sd and "p_values" not in sd:
        print(f"Skipping {model_name}, p_values/test-stats not computed")
        continue
    if wc["type"] == WatermarkType.KTH:
        print("kth")
        watermark_scores = sd["kth_test_stats"]
        null_scores = kth_ref_dist
    elif wc["type"] == WatermarkType.KGW:
        for name, ref_dist_data in ref_dist.items():
            ref_dist_wc = ref_dist_data["watermark_config"]
            if (
                ref_dist_wc["type"] == WatermarkType.KGW and 
                ref_dist_wc["gamma"] == wc["gamma"] and
                ref_dist_wc["seeding_scheme"] == wc["seeding_scheme"]
            ):
                print(name)
                watermark_scores = sd["p_values"]
                null_scores = ref_dist_data["p_values"]
                break
    elif wc["type"] == WatermarkType.AAR:
        for name, ref_dist_data in ref_dist.items():
            ref_dist_wc = ref_dist_data["watermark_config"]
            if (
                ref_dist_wc["type"] == WatermarkType.AAR and 
                ref_dist_wc["k"] == wc["k"]
            ):
                print(name)
                watermark_scores = sd["p_values"]
                null_scores = ref_dist_data["p_values"]
                break
    if watermark_scores is None:
        print(f"Skipping {model_name}, could not find ref dist for {wc}")
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
