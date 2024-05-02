## Generation and evaluation scripts

This subdirectory contains scripts for evaluating watermark distilled models.

### AUROC and KTH reference distributions

Before running any evaluations on a dataset, in order to compute AUROC values later, we need to compute a reference distribution of watermark detection p-values on human-generated text from that dataset. To do so, run the following from the top-level directory.
```
bash scripts/evaluate/auroc_ref_distribution.sh <dataset> <llama_path>
```
- `dataset` specifies the dataset. Supported datasets are [`c4`](https://huggingface.co/datasets/allenai/c4) (realnewslike), [`wikipedia`](https://huggingface.co/datasets/wikipedia), and [`arxiv`](https://huggingface.co/datasets/scientific_papers).
- `llama_path` (optional) specifies the path of the Llama 2 tokenizer. Defaults to [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf), which downloads from Hugging Face.

In order to compute KTH detection p-values later, we need to compute a reference distribution of KTH detection test statistics on human-generated text. To do, run the following.
```
bash scripts/evaluate/kth_ref_distribution.sh <dataset> <llama_path>
```
Note that KTH detection is relatively slow (several hours) and only requires CPU. If you do not wish to compute KTH detection p-values, you can skip this and comment out the KTH detection code in the following evaluation scripts. Since KTH detection only requires CPU, you can also comment it out in the following evaluation scripts and run it separately.

### Evaluate watermark distilled models

To generate and evaluate watermark distilled models, run the following.
```
bash scripts/evaluate/generate_and_evaluate.sh <dataset> <output_file> <llama_path> <perplexity_model> [models]...
```
- `dataset` specifies the dataset. Supported datasets are [`c4`](https://huggingface.co/datasets/allenai/c4) (realnewslike), [`wikipedia`](https://huggingface.co/datasets/wikipedia), and [`arxiv`](https://huggingface.co/datasets/scientific_papers).
- `output_file` specifies the output file.
- `llama_path` specifies the path of the Llama 2 tokenizer.
- `perplexity_model` specifies the model to use for computing perplexity (PPL). In the paper, we use [`meta-llama/Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf).
- `models` are the models to evaluate, separated by spaces. All models in one run should use the same tokenizer.

The batch sizes in the script are designed to fit on 1 NVIDIA A100 80GB GPU.

### Evaluate decoding-based watermarking

To evaluate decoding-based watermarking on Llama 2 7B or Pythia 1.4B, run either of the following.
```
bash scripts/evaluate/decoding_watermark_llama.sh <dataset> <output_file> <llama_path> <perplexity_model>
```
```
bash scripts/evaluate/decoding_watermark_pythia.sh <dataset> <output_file> <pythia_path> <perplexity_model>
```
The watermarking strategies that are used are taken from [`experiments/watermark-configs/watermark_configs_list.json`](/experiments/watermark-configs/watermark_configs_list.json).
