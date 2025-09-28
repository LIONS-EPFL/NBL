# Efficient Large Language Model Inference with Neural Block Linearization (NBL)

## Authors: Mete Erdogan, Francesco Tonin, Volkan Cevher

<p align="center">
  <img src="figures/nbl_overview.pdf" alt="Overview of Neural Block Linearization" width="500"/>
</p>

## Abstract
The high inference demands of transformer-based Large Language Models (LLMs) pose substantial challenges in their deployment. To this end, we introduce Neural Block Linearization (NBL), a novel framework for accelerating transformer model inference by replacing self-attention layers with linear approximations derived from Linear Minimum Mean Squared Error estimators. NBL leverages Canonical Correlation Analysis to compute a theoretical upper bound on the approximation error. Then, we use this bound as a criterion for substitution, selecting the LLM layers with the lowest linearization error. NBL can be efficiently applied to pre-trained LLMs without the need for fine-tuning. In experiments, NBL achieves notable computational speed-ups while preserving competitive accuracy on multiple reasoning benchmarks. For instance, applying NBL to 12 self-attention layers in DeepSeek-R1-Distill-Llama-8B increases the inference speed by 32\% with less than 1\% accuracy trade-off, making it a flexible and promising solution to improve the inference efficiency of LLMs.


---

## Quick Start

### Installation

```bash
conda create -n llm-drop python=3.10
conda activate llm-drop

# For NBL:
cd ./NBL
pip install -r requirements.txt
```

---

## Running NBL

#### Apply Attn NBL on Llama-3.1-8B
```bash
bash scripts/apply_nbl/layer_nbl_llama.sh
```

#### Apply Attn NBL on Mistral-7B
```bash
bash scripts/apply_nbl/layer_nbl_mistral.sh
```

These scripts will:
1. Generate importance scores for blocks/layers.
2. Determine which modules to retain/drop.
3. Save compressed model configs and weights.

- Intermediate outputs (CCA values, importance scores) → stored under `/llm_variables/`
- Compressed models → stored under `../results_prune/cache/`

---

## Benchmarks

### Reasoning Accuracy
Evaluate performance on reasoning benchmarks:

```bash
bash scripts/benchmark/benchmark_lm_eval_llama.sh
bash scripts/benchmark/benchmark_lm_eval_mistral.sh
```

- NBL evaluation builds on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- To reproduce results, please use [this fork](https://github.com/s1ghhh/lm-evaluation-harness).
- Custom modeling files for NBL-adapted Mistral/Llama are in `src/llmtuner/model`.

### Speedup Measurements
```bash
bash scripts/benchmark/benchmark_speed.sh
```

### Quantization
For AWQ-based quantization:

```bash
python quantize.py
```

See [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) for CUDA-specific installation details.

### Speculative Decoding (EAGLE + NBL)
```bash
bash Speculative/EAGLE/run_speculative_mt_bench.sh
```

### LoRA Fine-Tuning with NBL
```bash
cd LoRA
python lora.py        # trains LoRA adapters
python lora_save.py   # fuses tuned layers into NBL model
```

### Calibration Runtime (GPU Implementation)
```bash
cd "Calibration Runtime"
python calc.py
```

---

## Code Acknowledgements
- NBL builds on [CASE-Lab-UMD/LLM-Drop](https://github.com/CASE-Lab-UMD/LLM-Drop)
- Evaluation via [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Quantization via [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

---

## Citation
@article{erdogan2025efficient,
  title={Efficient Large Language Model Inference with Neural Block Linearization},
  author={Erdogan, Mete and Tonin, Francesco and Cevher, Volkan},
  journal={arXiv preprint arXiv:2505.21077},
  year={2025}
}
```
