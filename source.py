from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt

import torch

import pandas as pd

from pathlib import Path


def quantize_int(weight):
    max_int = 2 ** (int_bits - 1) - 1
    scale = weight.abs().max() / max_int

    return torch.round(weight / scale) * scale


def quantize_float(weight):
    # TODO: Implement float quantization

    raise NotImplementedError


def quantize_tensor(weight):
    return (quantize_int if q_type == "int" else quantize_float)(weight)


def quantize(layer):
    with torch.no_grad():
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight
                quantized = quantize_tensor(weight)
                weight.copy_(quantized)


case = 7

case_dir = Path(f"results/case{case:03d}")
case_dir.mkdir(parents=True)

metadata_file = case_dir / "metadata.txt"

prompts_dir = case_dir / "prompts"
prompts_dir.mkdir(parents=True, exist_ok=True)

original_type = (torch.float16, torch.float32, torch.float64)[1]

q_type = ("int", "float")[0]

int_bits = (2, 4, 8, 16, 32)[0]

float_sign_bits, float_exp_bits, float_mantissa_bits = 1, 8, 7

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-2b-it")[0]

tokenizer = AutoTokenizer.from_pretrained(model_name)

with open(metadata_file, "w") as f:
    f.write(f"Device: {device}\nModel: {model_name}\nOriginal type: {original_type}\nQuantization: {q_type}")

    if q_type == "int":
        f.write(f"{int_bits}\n")

    else:
        f.write(f"\nSign: {float_sign_bits}\nExponent: {float_exp_bits}\nMantissa: {float_mantissa_bits}\n")

prompts = ["Explain gravity.", "What is 173 × 29?", "Write a Python function to reverse a list.",
           "Translate 'Good morning' into Bulgarian.", "Why is the sky blue?"]

global_heatmap_mse = []
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=original_type).to(device)
model_q = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=original_type).to(device)
original_layers = [{k: v.clone() for k, v in layer.state_dict().items()} for layer in model_q.model.layers]

for i, prompt in enumerate(prompts):
    prompt_dir = prompts_dir / f"prompt{i:03d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_heatmap_mse = []

    with open(prompt_dir / "content.txt", "w", encoding="utf-8") as f:
        f.write(f"{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs_fp = model(**inputs, output_hidden_states=True)
    baseline_hidden = outputs_fp.hidden_states

    for j, layer in enumerate(model_q.model.layers):
        result = ""
        q_layer = prompt_dir / f"q_layer{j:03d}"
        q_layer.mkdir(exist_ok=True)

        damage_plot = {"layer": [], "mse": [], "cosine": []}

        try:
            quantize(layer)
            outputs_q = model_q(**inputs, output_hidden_states=True)

            for k, (fp, q) in enumerate(zip(baseline_hidden, outputs_q.hidden_states)):
                d = torch.mean((fp - q) ** 2).item()
                cos = torch.nn.functional.cosine_similarity(fp.flatten(), q.flatten(), 0).item()
                result += f"{k:02d}; MSE: {d:.4f}; cos: {cos:.4f}\n"

                damage_plot["layer"].append(k), damage_plot["mse"].append(d), damage_plot["cosine"].append(cos)
                prompt_heatmap_mse.append({"quantized_layer": j, "measured_layer": k, "mse": d})
                global_heatmap_mse.append({"prompt": prompt, "quantized_layer": j, "measured_layer": k, "mse": d})

            with open(q_layer / "results.txt", "w") as f:
                f.write(result)

        finally:
            layer.load_state_dict(original_layers[j])

        fig, ax1 = plt.subplots(figsize=(7, 4))

        ax1.plot(damage_plot["layer"], damage_plot["mse"], marker="o", color="tab:red")

        ax1.set_xlabel("Measured layer")
        ax1.set_ylabel("MSE", color="tab:red")

        ax2 = ax1.twinx()

        ax2.plot(damage_plot["layer"], damage_plot["cosine"], marker="s", color="tab:blue")

        ax2.set_ylabel("Cosine similarity", color="tab:blue")

        plt.title("Damage per layer")

        plt.grid(True)

        plt.savefig(q_layer / "damage_plot.png")

        plt.close()

    df = pd.DataFrame(prompt_heatmap_mse)

    pivot = df.pivot(index="quantized_layer", columns="measured_layer", values="mse")

    plt.figure(figsize=(10, 8))
    plt.imshow(pivot, aspect="auto", cmap="viridis")
    plt.xticks(range(len(pivot.columns)))
    plt.yticks(range(len(pivot.index)))
    plt.xlabel("Measured layer")
    plt.ylabel("Quantized layer")
    plt.colorbar(label="MSE")
    plt.savefig(prompt_dir / "heatmap_mse.png", bbox_inches="tight")

df = pd.DataFrame(global_heatmap_mse)
df = (df.groupby(["quantized_layer", "measured_layer"], as_index=False)[["mse"]].mean())

pivot = df.pivot(index="quantized_layer", columns="measured_layer", values="mse")

plt.figure(figsize=(10, 8))
plt.imshow(pivot, aspect="auto")
plt.xlabel("Measured layer")
plt.ylabel("Quantized layer")
plt.colorbar(label="MSE")
plt.savefig(case_dir / "heatmap_mse.png", bbox_inches="tight")
