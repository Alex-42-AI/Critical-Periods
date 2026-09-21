# LLM Layer-Wise Quantization Sensitivity

This project investigates how quantization affects the internal representations of large language models (LLMs), with particular focus on whether some transformer layers are substantially more sensitive or important than others.

The project contains two complementary experiments:

* **`source.py`**: quantizes one transformer layer at a time and measures how the resulting representation damage propagates through the model.
* **`reverse.py`**: quantizes the entire model and then restores one transformer layer at a time, measuring whether preserving a particular layer substantially reduces the resulting representation damage.

The experiments compare the quantized/hybrid models against an otherwise identical full-precision reference model using **mean absolute error (MAE)** and **cosine similarity** between hidden-state representations.

---

## Research question

The main research question is:

> **Are all transformer layers equally sensitive to low-bit quantization, or are some layers substantially more important to preserve?**

A particular hypothesis investigated by the project is:

> **Early transformer layers are particularly important to preserve when quantizing LLMs.**

The experiments are designed to test this at the level of internal representations rather than relying solely on final task accuracy.

---

# Methodology

## Model architecture

For a model with N transformer layers, the hidden states are treated as a sequence of representations:

$$
H_0 \rightarrow H_1 \rightarrow H_2 \rightarrow \cdots \rightarrow H_N
$$

where:

* \(H_0\) is the embedding representation;
* \(H_i\) corresponds to the representation after transformer layer \(i-1\);
* The final hidden-state representation includes the model's final normalization where applicable.

The experiments therefore use the hidden-state index as the **measured depth of the model**.

The final hidden-state behavior was explicitly verified for the investigated architectures using forward hooks to distinguish the final transformer-layer output from the subsequent final RMSNorm output.

---

# Quantization

The current implementation uses symmetric, per-layer linear quantization.

For a weight tensor \(W\), the maximum representable signed integer is

$$
q_{\max}=2^{b-1}-1
$$

where \(b\) is the selected number of bits.

The scale is calculated as

$$
s=\frac{\max(|W|)}{q_{\max}}
$$

and the quantized-and-dequantized weight is

$$
\hat W =
round\left(\frac{W}{s}\right)s.
$$

The implementation operates on `torch.nn.Linear` modules.

### Important implementation note

The current quantizer is a **simulated quantizer**. Quantized values are rounded to the selected quantization grid but are stored again in the original floating-point tensor.

Therefore:

* `int4` represents a 4-bit quantization **level**, not a physically packed 4-bit tensor.
* The experiments measure the effect of quantization error on model representations.
* They do **not** measure the actual memory usage or inference speed of a packed INT4 implementation.

This distinction is intentional for the current representation-sensitivity experiments.

---

# Metrics

Two metrics are used to compare hidden-state representations.

## Mean Absolute Error

For reference representation \(H\) and experimental representation \(\hat H\):

$$
\MAE
=
\frac{1}{n}
\sum_i |H_i-\hat H_i|.
$$

Lower MAE means the experimental representation is closer to the original model.

## Cosine similarity

The flattened hidden-state tensors are compared using cosine similarity:

$$
\cos(H,\hat H)
=
\frac{H\cdot\hat H}
{\|H\|\|\hat H\|}.
$$

A value closer to 1 indicates greater directional similarity.

Cosine similarity is calculated using FP64 inputs to reduce numerical error in the dot-product and norm reductions.

---

# `source.py`

## Purpose

`source.py` performs the **one-layer-at-a-time quantization experiment**.

For every selected model, quantization level and reference dtype, the script:

1. Loads the original model.
2. Runs the model on each evaluation prompt.
3. Stores the original hidden states.
4. Loads a second copy of the model.
5. Quantizes exactly one transformer layer.
6. Runs the model again.
7. Compares every resulting hidden state against the corresponding full-precision hidden state.
8. Restores the original layer.
9. Repeats for every transformer layer.

This produces a representation-damage profile for every possible quantized layer.

### Interpretation

For a particular quantized layer \(j\), the resulting curve shows:

> **How much does quantizing layer \(j\) affect representations at different depths of the model?**

The corresponding heatmap has:

* **x-axis**: measured hidden-state depth
* **y-axis**: quantized layer
* **cell value**: MAE or cosine similarity

Thus, each row represents the propagation of quantization damage caused by one particular layer.

---

# `reverse.py`

## Purpose

`reverse.py` performs the complementary **layer-sparing experiment**.

The model is first completely quantized.

For each transformer layer \(j\):

1. The fully quantized model is taken as the starting point.
2. Layer \(j\) is restored to its original full-precision weights.
3. The hybrid model is evaluated.
4. Its hidden states are compared against the original unquantized model.
5. The restored layer is quantized again.
6. The experiment continues with the next layer.

This creates a model of the form:

```text
Quantized + one spared layer
```

rather than:

```text
Full precision + one quantized layer
```

### Interpretation

For a particular spared layer \(j\), the resulting curve answers:

> **If all layers are quantized except layer \(j\), how much of the original representation is recovered?**

The heatmap therefore has:

* **x-axis**: measured hidden-state depth
* **y-axis**: spared layer
* **cell value**: MAE or cosine similarity between the unquantized and hybrid models

A particularly important layer should potentially produce a visibly different row because preserving it changes the model's trajectory more strongly than preserving an ordinary layer.

---

# Relationship between the two experiments

The two experiments answer related but different questions.

### `source.py`

> What happens when this particular layer is quantized?

```text
FP model
   |
   +-- quantize layer j
           |
           v
       hybrid model
           |
           v
   representation damage
```

### `reverse.py`

> What happens when this particular layer isn't quantized?

```text
FP model
   |
   +-- quantize every layer
           |
           +-- restore layer j
                    |
                    v
               hybrid model
                    |
                    v
            remaining damage
```

Using both experiments makes it possible to distinguish between a layer whose quantization causes unusually large damage and a layer whose preservation unusually reduces the damage of an otherwise fully quantized model.

---

# Experimental models

The current experiments include:

* `HuggingFaceTB/SmolLM2-360M`
* `HuggingFaceTB/SmolLM2-1.7B-Instruct`
* `HuggingFaceTB/SmolLM3-3B`
* `Qwen/Qwen2.5-3B`
* `Qwen/Qwen2.5-7B-Instruct`
* `microsoft/Phi-3-mini-4k-instruct`
* `mistralai/Mistral-7B-Instruct-v0.3`
* `meta-llama/Llama-3.2-1B`
* `meta-llama/Llama-3.2-3B`

`reverse.py` currently uses a subset of these models and additionally includes:

* `allenai/OLMo-1B-hf`
* `allenai/OLMo-7B-hf`

The exact model list is controlled directly in each script.

---

# Evaluation prompts

The current evaluation set consists of five prompts covering different types of language-model behavior:

```text
Explain gravity.

What is 173 × 29?

Write a Python function to reverse a list.

Translate 'Good morning' into Bulgarian.

Why is the sky blue?
```

The prompts are intentionally varied so that layer sensitivity can be examined across different types of inputs rather than a single task.

---

# Reference precision

The initial experiments investigate three floating-point reference types:

* FP16
* FP32
* FP64

This is used to determine whether the observed layer-wise quantization behavior is strongly dependent on the original model representation.

Subsequent experiments can use a single reference dtype once the initial precision-sensitivity experiment establishes that the qualitative results are sufficiently stable.

The reference dtype describes how the pretrained checkpoint is loaded. It does **not** mean that the model is randomly initialized at that precision.

---

# Results directory

Each experiment is assigned a numbered case:

```text
results/
├── case000/
├── case001/
├── case002/
└── ...
```

or, for the reverse experiment:

```text
reverse/
├── case000/
├── case001/
├── case002/
└── ...
```

Each case contains a `metadata.json` file describing the experimental conditions.

Example:

```text
case000/
├── metadata.json
├── global_results.json
├── fp_q_damage.png
├── fp_q_damage.pdf
├── heatmap_mae.png
├── heatmap_mae.pdf
├── heatmap_cos_sim.png
├── heatmap_cos_sim.pdf
└── prompts/
```

---

# Per-prompt results

Each prompt receives its own directory:

```text
prompts/
├── prompt0/
├── prompt1/
├── prompt2/
├── prompt3/
└── prompt4/
```

The original prompt is stored in:

```text
content.txt
```

This makes each experimental result self-contained and preserves the exact input used for the experiment.

---

# `source.py` output structure

For each quantized layer:

```text
prompt0/
└── q_layer003/
    ├── layer3_results.json
    ├── damage_plot_layer3.png
    └── damage_plot_layer3.pdf
```

The JSON contains the representation damage measured at each hidden-state depth.

At the prompt level:

```text
prompt0/
├── prompt0_results.json
├── heatmap_mae.png
├── heatmap_mae.pdf
├── heatmap_cos_sim.png
└── heatmap_cos_sim.pdf
```

The case-level results aggregate the measurements across all evaluation prompts.

---

# `reverse.py` output structure

The reverse experiment separates the layer-sparing results:

```text
prompt0/
├── prompt0_fp_q_damage.json
├── prompt0_fp_q_damage.png
├── prompt0_fp_q_damage.pdf
└── spared_layer003/
    ├── layer3_unquantized_vs_hybrid.json
    ├── layer3_damage.png
    └── layer3_damage.pdf
```

The prompt-level hybrid heatmaps are stored under:

```text
prompt0/
└── unquantized_vs_hybrid/
    ├── prompt0_results.json
    ├── heatmap_mae.png
    ├── heatmap_mae.pdf
    ├── heatmap_cos_sim.png
    └── heatmap_cos_sim.pdf
```

The case-level directory additionally contains the averaged results across all prompts.

---

# Reproducibility

Each case stores its experimental configuration in `metadata.json`, including:

* device
* model
* original floating-point type
* quantization precision
* evaluation prompts

The experimental scripts also use deterministic model weights from the specified Hugging Face checkpoints.

The generated result hierarchy is designed to preserve both the aggregate results and the individual measurements needed to reproduce or inspect the analysis.

---

# Installation

The project requires Python and the following main packages:

```bash
pip install torch transformers matplotlib pandas
```

A CUDA-capable GPU is recommended for larger models.

The scripts automatically select CUDA when available:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

# Running the experiments

Run the forward experiment with:

```bash
python source.py
```

Run the reverse layer-sparing experiment with:

```bash
python reverse.py
```

The `START` variable can be used to resume execution from a particular experiment case without modifying the experiment definitions.

---

# Research interpretation

The central analysis is not simply whether quantization increases error. Quantization is expected to introduce some error.

The important question is whether the error is **non-uniform across transformer depth**.

Evidence for layer sensitivity would appear as patterns such as:

* particular rows of the `source.py` heatmap showing substantially greater representation damage;
* particular rows of the `reverse.py` heatmap showing substantially lower error when that layer is spared;
* consistent patterns across different prompts;
* consistent patterns across different model architectures and sizes.

A layer should therefore not be considered important merely because one individual prompt produces a large value. Ideally, its behavior should be examined across prompts and, where possible, across models.

---

# Current hypothesis

The primary hypothesis is:

> **Initial transformer layers are particularly important to preserve during LLM quantization.**

The experiments are intended to test this hypothesis rather than assume it.

In particular, the project investigates whether early-layer sensitivity remains visible when:

* different prompts are used;
* different model architectures are evaluated;
* model size changes;
* quantization precision changes;
* the original floating-point precision changes.

A stronger conclusion requires consistent evidence across these conditions.

---

# Limitations

Several limitations should be kept in mind when interpreting the results.

### Simulated quantization

The current quantizer does not pack weights into actual INT2/INT4 storage. It measures the effect of quantization error, not the practical memory or inference-speed benefits of a real low-bit implementation.

### Limited evaluation prompts

Only a small number of prompts are currently used. The results therefore measure representation sensitivity on the selected inputs rather than universal model behavior.

### Representation metrics

MAE and cosine similarity measure differences between hidden-state tensors, but they do not directly measure downstream task performance or generated-text quality.

### Model-specific architecture

Layer numbering and final normalization behavior can differ between model architectures. The interpretation of hidden-state indices must therefore take the specific architecture into account.

### Quantization method

The conclusions apply directly to the particular symmetric per-layer quantization procedure used here. Other quantization schemes may produce different sensitivity patterns.

---

# Project structure

```text
.
├── source.py
├── reverse.py
├── results/
│   └── case*/
└── reverse/
    └── case*/
```

The two scripts intentionally share much of their experimental infrastructure while investigating opposite directions of the same question:

```text
source.py
    "What happens if I quantize this layer?"

reverse.py
    "What happens if I protect this layer?"
```

Together, they provide a layer-wise view of how quantization errors affect the internal representation trajectory of LLMs.
