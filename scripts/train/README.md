## Watermark distillation training scripts

This subdirectory contains scripts for training using logit-based and sampling-based watermark distillation.

### Logit-based watermark distillation

[`train_llama_logit_distill.sh`](train_llama_logit_distill.sh) runs logit-based watermark distillation on Llama 2 7B. The training configuration is for 4 NVIDIA A100 80GB GPUs. The script is run from the top-level directory as
```
bash scripts/train/train_llama_logit_distill.sh <watermark_type> <output_dir/> <master_port> <llama_path>
```
- `watermark_type` specifies the watermarking strategy for training. The possible types are listed at the [end](#watermark-types) of this README.
- `output_dir` specifies the directory where the model should be stored (with the trailing `/`). This should not include the model name itself, which is automatically computed by the script.
- `master_port` is the port that is passed to `torchrun`. This can be more or less arbitrarily selected.
- `llama_path` (optional) specifies the path where the base Llama 2 7B model weights are loaded from. Defaults to [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf), which downloads from Hugging Face.

### Sampling-based watermark distillation

To perform sampling-based watermark distillation, first, `generate_sampling_distill_train_data.sh` generates watermarked samples from the teacher Llama 2 7B to use as training data. We used 1 NVIDIA A100 80GB GPU. The script is run from the top-level directory as 
```
bash scripts/train/generate_sampling_distill_train_data.sh <watermark_type> <llama_path>
```
- `watermark_type` specifies the watermarking strategy for training. The possible types are listed at the [end](#watermark-types) of this README.
- `llama_path` (optional) specifies the path where the base Llama 2 7B model weights are loaded from. Defaults to [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf), which downloads from Hugging Face.

Then, to run sampling-based watermark distillation on Llama 2 7B as the student (on 4 A100 NVIDIA 80GB GPUs), the script is run as
```
bash scripts/train/train_llama_sampling_distill.sh <watermark_type> <output_dir/> <master_port> <llama_path>
```
- `watermark_type` specifies the watermarking strategy for training. The possible types are listed at the [end](#watermark-types) of this README.
- `output_dir` specifies the directory where the model should be stored (with the trailing `/`). This should not include the model name itself, which is automatically computed by the script.
- `master_port` is the port that is passed to `torchrun`. This can be more or less arbitrarily selected.
- `llama_path` (optional) specifies the path where the base Llama 2 7B model weights are loaded from. Defaults to [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf), which downloads from Hugging Face.

To run sampling-based watermark distillation on Pythia 1.4B as the student (on 1 A100 NVIDIA 80GB GPU), the script is similarly run as
```
bash scripts/train/train_pythia_sampling_distill.sh <watermark_type> <output_dir/> <master_port> <pythia_path>
```
- `pythia_path` (optional) specifies the path where the base Pythia 1.4B model weights are loaded from. Defaults to [`EleutherAI/pythia-1.4b`](https://huggingface.co/EleutherAI/pythia-1.4b), which downloads from Hugging Face.

### Watermark types

These are the strings that can be passed into the training scripts to specify the watermark type. The watermark configuration files are in [`experiments/watermark-configs`](/experiments/watermark-configs).

KGW 
- `kgw-k0-gamma0.25-delta1`
- `kgw-k0-gamma0.25-delta2`
- `kgw-k1-gamma0.25-delta1`
- `kgw-k1-gamma0.25-delta2`
- `kgw-k2-gamma0.25-delta2`

Aar
- `aar-k2`
- `aar-k3`
- `aar-k4`

KTH
- `kth-shift1`
- `kth-shift2`
- `kth-shift4`
- `kth-shift256`
