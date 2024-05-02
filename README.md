# On the Learnability of Watermarks for Language Models

This repository contains code for the paper [On the Learnability of Watermarks for Language Models](https://arxiv.org/abs/2312.04469) by Chenchen Gu, Xiang Lisa Li, Percy Liang, and Tatsunori Hashimoto.

### Setup

To install the necessary packages, first create a conda environment.
```
conda create -n <env_name> python=3.11
conda activate <env_name>
```
Then, install the required packages with 
```
pip install -r requirements.txt
```

### Usage

We include scripts for reproducing experiments in the paper in the [`scripts`](scripts) directory, which also serve as examples for how to run the files in this repository. `README.md`'s within [`scripts`](scripts) provide instructions on how to run the scripts. Note that all scripts should be run from the top-level directory.

Feel free to create an issue if you encounter any problems or bugs!

### References

Code in the [`watermarks/kgw`](watermarks/kgw) directory is from [github.com/jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking). In the [`watermarks/kth`](watermarks/kth) directory, `detect.py`, `levenshtein.pyx`, and `mersenne.py` are from [github.com/jthickstun/watermark](https://github.com/jthickstun/watermark). [`train_logit_distill.py`](train_logit_distill.py) and [`train_sampling_distill.py`](train_sampling_distill.py) are adapted from [github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py).

## Models

Below are links to trained model weights from the paper's experiments (hosted on Hugging Face).

### Logit-based watermark distilled Llama 2 7B

- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta1)$k = 0, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k0-gamma0.25-delta2)$k = 0, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta1)$k = 1, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2)$k = 1, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kgw-k2-gamma0.25-delta2)$k = 2, \gamma = 0.25, \delta = 2$
- [Aar k = 2](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/llama-2-7b-logit-watermark-distill-kth-shift256)

### Sampling-based watermark distilled Llama 2 7B

- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k0-gamma0.25-delta1)$k = 0, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k0-gamma0.25-delta2)$k = 0, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k1-gamma0.25-delta1)$k = 1, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2)$k = 1, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-k2-gamma0.25-delta2)$k = 2, \gamma = 0.25, \delta = 2$
- [Aar k = 2](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift256)

### Sampling-based watermark distilled Pythia 1.4B

- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k0-gamma0.25-delta1)$k = 0, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k0-gamma0.25-delta2)$k = 0, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta1)$k = 1, \gamma = 0.25, \delta = 1$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k1-gamma0.25-delta2)$k = 1, \gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-k2-gamma0.25-delta2)$k = 2, \gamma = 0.25, \delta = 2$
- [Aar k = 2](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift256)
