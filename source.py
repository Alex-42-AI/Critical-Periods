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

prompts_dir = case_dir / "prompts"

metadata_file = case_dir / "metadata.txt"

original_type = torch.float16

q_type = torch.int8

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = ["HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-3B", "meta-llama/Llama-3.1-8B-Instruct"][0]

tokenizer = AutoTokenizer.from_pretrained(model_name)

with open(metadata_file, "w") as f:
    f.write(f"Device: {device}\nModel: {model_name}\nOriginal type: {original_type}\nQuantization: {q_type}\n")

prompts = [
    "Explain gravity.",
    "What is 173 × 29?",
    "Write a Python function to reverse a list.",
    "Translate 'Good morning' into Bulgarian.",
    "Why is the sky blue?"
]

prompt_results, results = [], []

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=original_type
)
model = model.to(device)

for i, prompt in enumerate(prompts):
    curr_prompt = prompts_dir / f"prompt{i}"

    with open(curr_prompt / "content.txt") as f:
        f.write(prompt)

    result = ""
    inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs_fp = model(
        **inputs,
        output_hidden_states=True
    )
    baseline_hidden = outputs_fp.hidden_states

    for j in range(len(model.model.layers)):
        result += f"Quantized layer {j}:\n"
        model_q = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=original_type
        )
        model_q = model_q.to(device)
        quantize(model_q.model.layers[j])
        outputs_q = model_q(
            **inputs,
            output_hidden_states=True
        )

        for k, (fp, q) in enumerate(zip(baseline_hidden, outputs_q.hidden_states)):
            d = damage(fp, q)
            cos = torch.nn.functional.cosine_similarity(
                fp.flatten(),
                q.flatten(),
                dim=0
            ).item()
            result += f"{k:02d}; MSE: {d:.4f}; cos: {cos:.4f}\n"

            results.append({
                "prompt": prompt,
                "quantized_layer": j,
                "measured_layer": k,
                "mse": d,
                "cosine": cos
            })

        result += "\n"
        prompt_results.append(result)

df = pd.DataFrame(results)
df = (
    df.groupby(
        ["quantized_layer", "measured_layer"],
        as_index=False
    )[["mse", "cosine"]]
    .mean()
)

pivot = df.pivot(
    index="quantized_layer",
    columns="measured_layer",
    values="mse"
)

plt.figure(figsize=(10, 8))
plt.imshow(pivot, aspect="auto")
plt.xlabel("Measured layer")
plt.ylabel("Quantized layer")
plt.colorbar(label="MSE")
plt.savefig(
    case_dir / "heatmap_mse.png",
    bbox_inches="tight"
)

for j, (prompt, result) in enumerate(zip(prompts, prompt_results)):
    with open(prompts_dir / f"prompt{j}.txt", "w") as f:
        f.write(prompt + "\n" + result + "\n")

for prompt in df["prompt"].unique():
    for layer in df["quantized_layer"].unique():
        subset = df[df["quantized_layer"] == layer]

        plt.figure(figsize=(6, 4))

        plt.plot(
            subset["measured_layer"],
            subset["mse"],
            marker="o"
        )

        plt.title(f"Quantized layer {layer}")
        plt.xlabel("Measured layer")
        plt.ylabel("Damage")

        plt.grid(True)
        plt.savefig(case_dir / f"layer{layer:02d}.png", bbox_inches="tight")
