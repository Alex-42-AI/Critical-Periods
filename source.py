from gc import collect

from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt

import torch

from pandas import DataFrame

from pathlib import Path


def quantize_tensor(weight):
    max_int = 2 ** (q_bits - 1) - 1
    scale = weight.abs().max() / max_int

    return torch.round(weight / scale) * scale


def quantize(layer):
    with torch.no_grad():
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight
                quantized = quantize_tensor(weight)
                weight.copy_(quantized)


device = "cuda" if (p := torch.cuda.is_available()) else "cpu"

# original_type = (torch.float16, torch.float32, torch.float64)[2]
#
# q_bits = (2, 4, 8, 16, 32)[0]
#
# model_name = ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM3-3B",
#               "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3-mini-4k-instruct",
#               "mistralai/Mistral-7B-Instruct-v0.3")[6]

prompts = ["Explain gravity.", "What is 173 × 29?", "Write a Python function to reverse a list.",
           "Translate 'Good morning' into Bulgarian.", "Why is the sky blue?"]

case = 72
experiments = []

for i, original_type in enumerate([torch.float16, torch.float32, torch.float64]):
    for q_bits in [32, 16, 8, 4, 2][2 - i:]:
        experiments.append(("mistralai/Mistral-7B-Instruct-v0.3", original_type, q_bits))

for model_name, original_type, q_bits in experiments:
    case_dir = Path(f"results/case{case:03d}")
    case_dir.mkdir(parents=True)
    case += 1

    metadata_file = case_dir / "metadata.txt"

    prompts_dir = case_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "w") as f:
        f.write(f"Device: {device}\nModel: {model_name}\nOriginal type: {original_type}\nQuantization: int{q_bits}\n")

    global_heatmap_mse = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, prompt in enumerate(prompts):
        prompt_dir = prompts_dir / f"prompt{i:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_heatmap_mse = []

        with open(prompt_dir / "content.txt", "w", encoding="utf-8") as f:
            f.write(f"{prompt}\n")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)
        model.eval()

        with torch.inference_mode():
            outputs_fp = model(**inputs, output_hidden_states=True)
            baseline_hidden = outputs_fp.hidden_states

        del model, outputs_fp

        model_q = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)
        model_q.eval()

        for j, layer in enumerate(model_q.model.layers[:-1]):
            result = []
            q_layer = prompt_dir / f"q_layer{j:03d}"
            q_layer.mkdir(exist_ok=True)
            damage_plot = {"layer": [], "mse": [], "cosine": []}

            with torch.no_grad():
                restore_layer = {k: v.clone() for k, v in layer.state_dict().items()}

            try:
                quantize(layer)

                with torch.inference_mode():
                    outputs_q = model_q(**inputs, output_hidden_states=True)

                for k, (fp, q) in enumerate(zip(baseline_hidden, outputs_q.hidden_states)):
                    d = torch.mean((fp - q) ** 2).item()
                    cos = torch.nn.functional.cosine_similarity(fp.flatten(), q.flatten(), 0).item()
                    result.append(f"{k:02d}; MSE: {d:.4f}; cos: {cos:.4f}\n")

                    damage_plot["layer"].append(k), damage_plot["mse"].append(d), damage_plot["cosine"].append(cos)
                    prompt_heatmap_mse.append({"quantized_layer": j, "measured_layer": k, "mse": d})
                    global_heatmap_mse.append({"prompt": prompt, "quantized_layer": j, "measured_layer": k, "mse": d})

                with open(q_layer / "results.txt", "w") as f:
                    f.write("".join(result))

            finally:
                layer.load_state_dict(restore_layer)

                del restore_layer

                try:
                    del outputs_q

                except NameError:
                    ...

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

            del damage_plot

        del model_q, baseline_hidden, inputs

        collect()

        if p:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        df = DataFrame(prompt_heatmap_mse)

        pivot = df.pivot(index="quantized_layer", columns="measured_layer", values="mse")

        plt.figure(figsize=(10, 8))
        plt.imshow(pivot, aspect="auto", cmap="viridis")
        plt.xticks(range(len(pivot.columns)))
        plt.yticks(range(len(pivot.index)))
        plt.xlabel("Measured layer")
        plt.ylabel("Quantized layer")
        plt.colorbar(label="MSE")
        plt.savefig(prompt_dir / "heatmap_mse.png", bbox_inches="tight")

    del tokenizer

    df = DataFrame(global_heatmap_mse)
    df = (df.groupby(["quantized_layer", "measured_layer"], as_index=False)[["mse"]].mean())

    pivot = df.pivot(index="quantized_layer", columns="measured_layer", values="mse")

    plt.figure(figsize=(10, 8))
    plt.imshow(pivot, aspect="auto")
    plt.xlabel("Measured layer")
    plt.ylabel("Quantized layer")
    plt.colorbar(label="MSE")
    plt.savefig(case_dir / "heatmap_mse.png", bbox_inches="tight")
