from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt

import torch

import pandas as pd

from pathlib import Path


def damage(hidden_fp, hidden_q):
    return torch.mean((hidden_fp - hidden_q) ** 2).item()


def quantize_tensor(weight):
    scale = weight.abs().max() / torch.iinfo(q_type).max
    return torch.round(weight / scale) * scale


def quantize(layer):
    with torch.no_grad():
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight
                quantized = quantize_tensor(weight)
                weight.copy_(quantized)


case = 0

case_dir = Path(f"results/case{case:03d}")
case_dir.mkdir(parents=True)

metadata_file = case_dir / "metadata.txt"

prompts_dir = case_dir / "prompts"
prompts_dir.mkdir(parents=True, exist_ok=True)

original_type = torch.float16

q_type = torch.int8

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = ["HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-3B", "meta-llama/Llama-3.1-8B-Instruct"][0]

tokenizer = AutoTokenizer.from_pretrained(model_name)

with open(metadata_file, "w") as f:
    f.write(f"Device: {device}\nModel: {model_name}\nOriginal type: {original_type}\nQuantization: {q_type}\n")

prompts = ["Explain gravity.", "What is 173 × 29?", "Write a Python function to reverse a list.",
           "Translate 'Good morning' into Bulgarian.", "Why is the sky blue?"]

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=original_type)
model = model.to(device)
global_heatmap_mse = []

for i, prompt in enumerate(prompts):
    prompt_dir = prompts_dir / f"prompt{i:03d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_heatmap_mse = []

    with open(prompt_dir / "content.txt", "w") as f:
        f.write(f"{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs_fp = model(**inputs, output_hidden_states=True)
    baseline_hidden = outputs_fp.hidden_states

    for j in range(len(model.model.layers)):
        result = ""
        q_layer = prompt_dir / f"q_layer{j:03d}"
        q_layer.mkdir(exist_ok=True)

        damage_plot = {"layer": [], "mse": [], "cosine": []}

        model_q = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=original_type)
        model_q = model_q.to(device)
        quantize(model_q.model.layers[j])
        outputs_q = model_q(**inputs, output_hidden_states=True)

        for k, (fp, q) in enumerate(zip(baseline_hidden, outputs_q.hidden_states)):
            d = damage(fp, q)
            cos = torch.nn.functional.cosine_similarity(fp.flatten(), q.flatten(), 0).item()
            result += f"{k:02d}; MSE: {d:.4f}; cos: {cos:.4f}\n"

            damage_plot["layer"].append(k), damage_plot["mse"].append(d), damage_plot["cosine"].append(cos)
            prompt_heatmap_mse.append({"quantized_layer": j, "measured_layer": k, "mse": d})
            global_heatmap_mse.append({"prompt": prompt, "quantized_layer": j, "measured_layer": k, "mse": d})

        with open(q_layer / "results.txt", "w") as f:
            f.write(result)

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
    plt.imshow(pivot, aspect="auto")
    plt.xlabel("Measured layer")
    plt.ylabel("Quantized layer")
    plt.colorbar(label="MSE")
    plt.savefig(prompt_dir / "heatmap_mse.png", bbox_inches="tight")

df = pd.DataFrame(global_heatmap_mse)
df = (
    df.groupby(
        ["quantized_layer", "measured_layer"],
        as_index=False
    )[["mse"]]
    .mean()
)

pivot = df.pivot(index="quantized_layer", columns="measured_layer", values="mse")

plt.figure(figsize=(10, 8))
plt.imshow(pivot, aspect="auto")
plt.xlabel("Measured layer")
plt.ylabel("Quantized layer")
plt.colorbar(label="MSE")
plt.savefig(case_dir / "heatmap_mse.png", bbox_inches="tight")
