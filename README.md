# On the Learnability of Watermarks for Language Models

This repository contains code for the paper [On the Learnability of Watermarks for Language Models](https://arxiv.org/abs/2312.04469) by Chenchen Gu, Xiang Lisa Li, Percy Liang, and Tatsunori Hashimoto.

The `kgw_watermarking` directory is from [github.com/jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking). In the `kth_watermarking` directory, `detect.py`, `levenshtein.pyx`, and `mersenne.py` are from [github.com/jthickstun/watermark](https://github.com/jthickstun/watermark). `train_logits_distill.py` and `train_sampling_distill.py` are adapted from [github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py).

Below are links to trained model weights from the paper's experiments.

### Logits-based watermark distilled Llama 2 7B

- [KGW ](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-kgw-gamma0.25-delta2)$\gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-kgw-gamma0.25-delta1)$\gamma = 0.25, \delta = 1$
- [Aar k = 2](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/llama-2-7b-logits-watermark-distill-kth-shift256)

### Sampling-based watermark distilled Llama 2 7B

- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-gamma0.25-delta2)$\gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kgw-gamma0.25-delta1)$\gamma = 0.25, \delta = 1$
- [Aar k = 2](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/llama-2-7b-sampling-watermark-distill-kth-shift256)

### Sampling-based watermark distilled Pythia 1.4B

- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-gamma0.25-delta2)$\gamma = 0.25, \delta = 2$
- [KGW ](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kgw-gamma0.25-delta1)$\gamma = 0.25, \delta = 1$
- [Aar k = 2](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k2)
- [Aar k = 3](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k3)
- [Aar k = 4](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-aar-k4)
- [KTH s = 1](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift1)
- [KTH s = 2](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift2)
- [KTH s = 4](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift4)
- [KTH s = 256](https://huggingface.co/cygu/pythia-1.4b-sampling-watermark-distill-kth-shift256)
