# Critical Periods

An experimental study of how quantization-induced representation damage propagates through the layers of transformer-based language models.

## Overview

This project investigates how quantizing an individual transformer layer affects the hidden-state representations produced by subsequent layers.

For each experiment, a pretrained causal language model is loaded in a specified floating-point type. The weights of one transformer layer are then quantized using simulated symmetric integer quantization, while the remaining layers retain their original weights. The quantized model is evaluated on several prompts and its hidden states are compared with those of the unmodified model.

The goal is to examine:

* How much representation damage is introduced by quantizing a particular layer.
* How that damage propagates through subsequent transformer layers.
* Whether the effects of quantization differ between models, numerical precisions and quantization levels.
* How different prompts affect the observed representation damage.

## Models

The experiments currently use:

* `HuggingFaceTB/SmolLM2-360M`
* `HuggingFaceTB/SmolLM2-1.7B-Instruct`
* `HuggingFaceTB/SmolLM3-3B`
* `Qwen/Qwen2.5-3B`
* `Qwen/Qwen2.5-7B-Instruct`
* `microsoft/Phi-3-mini-4k-instruct`
* `mistralai/Mistral-7B-Instruct-v0.3`

## Prompts

Five prompts are used to provide different types of input:

1. `Explain gravity.`
2. `What is 173 × 29?`
3. `Write a Python function to reverse a list.`
4. `Translate 'Good morning' into Bulgarian.`
5. `Why is the sky blue?`

## Experimental Variables

### Original floating-point type

Experiments are performed using:

* `float16`
* `float32`
* `float64`

Quantization is only implemented from a floating point type to an integer type with fewer bits.

### Quantization level

The simulated integer quantization levels are:

* `int32`
* `int16`
* `int8`
* `int4`
* `int2`

Quantization uses symmetric per-tensor scaling based on the maximum absolute weight value.

For a weight tensor (W), the scale is calculated as:

[
s = \frac{\max(|W|)}{2^{b-1}-1}
]

where (b) is the number of quantization bits.

The quantized-and-dequantized weights are then:

[
\hat{W} = \operatorname{round}\left(\frac{W}{s}\right)s
]

The experiment therefore simulates integer quantization while keeping the model weights represented as floating-point tensors during inference.

## Layer-wise Experiment

For each model and prompt:

1. Run the unmodified model and save its hidden states as the baseline.
2. Load a second copy of the model.
3. Select one transformer layer.
4. Quantize all `Linear` weights within that layer.
5. Run the quantized model on the same prompt.
6. Compare the baseline and quantized hidden states.
7. Restore the original weights.
8. Repeat for the next transformer layer.

The final transformer layer is not quantized. Since there are no subsequent transformer layers through which its damage could propagate, quantizing it would represent a different experiment.

## Representation Damage Metrics

Two metrics are recorded.

### Mean Absolute Error

[
MAE = \frac{1}{N}\sum_{i=1}^{N}|x_i-y_i|
]

MAE measures the average absolute difference between the baseline and quantized hidden-state tensors.

### Cosine Similarity

[
\cos(x,y)=\frac{x\cdot y}{|x||y|}
]

Cosine similarity measures the directional similarity between the flattened baseline and quantized hidden-state tensors.

The metric calculations are performed using `float32` representations to reduce numerical issues when the underlying model uses lower-precision tensors.

## Final RMSNorm

The models also apply a final RMSNorm after the transformer blocks.

The output of this normalization is returned as the final hidden-state entry by the model, but it is **not treated as an additional transformer layer** in the main layer-wise analysis.

The final RMSNorm representation is therefore stored separately.

This distinction was made after observing that the MAE can drop dramatically between the output of the final transformer block and the final normalized representation. For example, a representation can exhibit substantial damage immediately after the final transformer block while showing much smaller MAE after the terminal RMSNorm.

The main heatmaps and layer-wise plots therefore measure transformer-layer outputs only, while the RMSNorm results are retained separately for additional analysis.

## Output Structure

Each experimental configuration is stored in its own directory:

```text
results/
└── case{i}/
    ├── metadata.txt
    ├── global_results.json
    ├── global_RMSNorm.json
    ├── heatmap_mae.png
    ├── heatmap_mae.pdf
    └── prompts/
        └── prompt{j}/
            ├── content.txt
            ├── prompt{j}_results.json
            ├── prompt{j}_RMSNorm.json
            ├── prompt{j}_heatmap_mae.png
            ├── prompt{j}_heatmap_mae.pdf
            └── q_layer{k}/
                ├── layer{k}_results.json
                ├── layer{k}_RMSNorm.json
                ├── damage_plot_layer{k}.png
                └── damage_plot_layer{k}.pdf
```

### `metadata.txt`

Contains the experimental configuration:

* device
* model
* original floating-point type
* quantization level

### `global_results.json`

Contains the MAE and cosine similarity measurements for every prompt, quantized layer and measured transformer layer.

### `global_RMSNorm.json`

Contains the corresponding measurements for the final RMSNorm output.

### `*_results.json`

Contains layer-wise representation-damage measurements for individual prompts or quantized layers.

### `*_RMSNorm.json`

Contains the corresponding final RMSNorm measurements.

### Heatmaps

Heatmaps show the mean MAE for each combination of:

* quantized layer
* measured transformer layer

The horizontal axis represents the measured layer, while the vertical axis represents the layer that was quantized.

### Damage plots

Individual plots show how MAE and cosine similarity change across measured transformer layers after quantizing a particular layer.

## Interpreting the Heatmap

A single row represents the propagation of damage caused by quantizing one transformer layer.

For example:

```text
                Measured layer →
Quantized    ┌─────────────────────
layer        │  damage propagation
     ↓       │
             │
```

This makes it possible to investigate whether damage:

* remains localized,
* propagates through later layers,
* increases with depth,
* diminishes after subsequent processing,
* or exhibits model- or prompt-specific behavior.

## Experimental Verification

The relationship between the final hidden state and the terminal RMSNorm was explicitly verified using forward hooks on the final transformer layer and the model's `norm` module.

This verification established that the final hidden-state representation corresponds to the output of the model's final RMSNorm rather than representing an additional transformer layer.

## Reproducibility

Each experiment records its configuration and preserves raw measurements in JSON format. The generated plots are retained alongside the underlying data so that results can be inspected and processed independently of the visualization code.

The experiments are designed to compare models and numerical configurations under the same set of prompts and layer-wise quantization procedure.
