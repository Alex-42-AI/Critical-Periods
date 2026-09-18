from gc import collect

from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt

import torch

from pandas import DataFrame

from pathlib import Path

from json import dump


def quantize_tensor(weight):
    max_int = 2 ** (q_bits - 1) - 1

    if not max_int:
        return weight.clone()

    scale = weight.abs().max() / max_int

    return torch.round(weight / scale) * scale


def quantize(lyr):
    with torch.no_grad():
        for module in lyr.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight
                quantized = quantize_tensor(weight)
                weight.copy_(quantized)


def plot_heatmap(dataframe, index, columns, value, xlabel, ylabel, title, *output_files):
    pivot = dataframe.pivot(index=index, columns=columns, values=value)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(pivot, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.colorbar(image, ax=ax, label=value.upper())
    fig.tight_layout()

    for output_file in output_files:
        plt.savefig(output_file, bbox_inches="tight")

    plt.close(fig)


device = "cuda" if (p := torch.cuda.is_available()) else "cpu"

original_types = (torch.float16, torch.float32, torch.float64)

q_bits_ls = (32, 16, 8, 4, 2)

model_names = ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM3-3B",
               "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3-mini-4k-instruct",
               "mistralai/Mistral-7B-Instruct-v0.3")

prompts = ["Explain gravity.", "What is 173 × 29?", "Write a Python function to reverse a list.", "Translate 'Good morning' into Bulgarian.", "Why is the sky blue?"]

START = 0
experiments = []

for model_name in model_names:
    for i, q_bits in enumerate(q_bits_ls):
        for original_type in original_types[max(0, 2 - i):]:
            experiments.append((model_name, original_type, q_bits))

for case, (model_name, original_type, q_bits) in enumerate(experiments[START:], START):
    print(case, model_name, original_type, q_bits)

    case_dir = Path(f"results/case{case:03d}")
    case_dir.mkdir(parents=True)

    prompts_dir = case_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    with open(case_dir / "metadata.json", "w") as f:
        dump({"Device": device, "Model": model_name, "Original type": str(original_type)[6:], "Quantization": f"int{q_bits}", "Prompts": prompts}, f, indent=4)

    global_heatmap, global_result_json = [], []

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, prompt in enumerate(prompts):
        print(prompt)

        prompt_dir = prompts_dir / f"prompt{i}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_heatmap, prompt_result_json = [], []

        with open(prompt_dir / "content.txt", "w", encoding="utf-8") as f:
            f.write(f"{prompt}\n")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        unquantized = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)
        unquantized.eval()

        # Verification experiment:
        # hidden_states[-1] corresponds to the output of the model's final RMSNorm,
        # rather than an additional transformer layer. This was verified by comparing
        # the final transformer-layer and RMSNorm outputs using forward hooks.

        with torch.inference_mode():
            outputs_unquantized = unquantized(**inputs, output_hidden_states=True)
            unquantized_hidden = outputs_unquantized.hidden_states

        del unquantized, outputs_unquantized

        quantized = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)
        quantized.eval()

        for j, layer in enumerate(quantized.model.layers):
            layer_result_json = []
            q_layer = prompt_dir / f"q_layer{j:03d}"
            q_layer.mkdir(exist_ok=True)
            damage_plot = {"layer": [], "mae": [], "cosine": []}

            with torch.no_grad():
                restore_layer = {k: v.clone() for k, v in layer.state_dict().items()}

            try:
                quantize(layer)

                with torch.inference_mode():
                    outputs_quantized = quantized(**inputs, output_hidden_states=True)
                    quantized_hidden = outputs_quantized.hidden_states

                for k, (fp, q) in enumerate(list(zip(unquantized_hidden, quantized_hidden))[1:], 1):
                    mae = torch.mean(torch.abs(fp.float() - q.float())).item()
                    cos = torch.nn.functional.cosine_similarity(fp.double().flatten(), q.double().flatten(), 0).item()

                    damage_plot["layer"].append(k), damage_plot["mae"].append(mae), damage_plot["cosine"].append(cos)

                    prompt_heatmap.append({"quantized layer": j, "measured hidden": k, "mae": mae, "cosine": cos})
                    global_heatmap.append({"prompt": prompt, "quantized layer": j, "measured hidden": k, "mae": mae, "cosine": cos})

                    layer_result_json.append({"measured hidden": k, "mae": mae, "cos": cos})
                    prompt_result_json.append({"quantized layer": j, "measured hidden": k, "mae": mae, "cos": cos})
                    global_result_json.append({"prompt": prompt, "quantized layer": j, "measured hidden": k, "mae": mae, "cos": cos})

                with open(q_layer / f"layer{j}_results.json", "w") as f:
                    dump(layer_result_json, f, indent=4)

                    del layer_result_json

            finally:
                layer.load_state_dict(restore_layer)

                del restore_layer, outputs_quantized

            fig, ax1 = plt.subplots(figsize=(9, 4))

            ax1.plot(damage_plot["layer"], damage_plot["mae"], marker="o", color="tab:red")
            ax1.set_xlabel("Measured hidden")
            ax1.set_ylabel("MAE", color="tab:red")

            ax2 = ax1.twinx()
            ax2.plot(damage_plot["layer"], damage_plot["cosine"], marker="s", color="tab:blue")
            ax2.set_ylabel("Cosine similarity", color="tab:blue")

            plt.title(f"{Path(model_name).name} | Quantized layer {j} | {str(original_type)[6:]} → int{q_bits}")
            plt.grid(True)

            plt.savefig(q_layer / f"damage_plot_layer{j}.png")
            plt.savefig(q_layer / f"damage_plot_layer{j}.pdf")
            plt.close()

            del damage_plot

        del quantized, unquantized_hidden, inputs

        collect()

        if p:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with open(prompt_dir / f"prompt{i}_results.json", "w") as f:
            dump(prompt_result_json, f, indent=4)

            del prompt_result_json

        df = DataFrame(prompt_heatmap)
        plot_heatmap(df, "quantized layer", "measured hidden", "mae", "Measured hidden", "Quantized layer", f"{Path(model_name).name}\nPrompt {i}: MAE representation damage | {str(original_type)[6:]} → int{q_bits}", prompt_dir / "heatmap_mae.png", prompt_dir / "heatmap_mae.pdf")

        df = DataFrame(prompt_heatmap)
        plot_heatmap(df, "quantized layer", "measured hidden", "cosine", "Measured hidden", "Quantized layer", f"{Path(model_name).name}\nPrompt {i}: cos sim representation damage | {str(original_type)[6:]} → int{q_bits}", prompt_dir / "heatmap_cos_sim.png", prompt_dir / "heatmap_cos_sim.pdf")

    del tokenizer

    with open(case_dir / "global_results.json", "w") as f:
        dump(global_result_json, f, indent=4)

        del global_result_json

    df = DataFrame(global_heatmap)
    df = df.groupby(["quantized layer", "measured hidden"], as_index=False)[["mae"]].mean()
    plot_heatmap(df, "quantized layer", "measured hidden", "mae", "Measured hidden", "Quantized layer", f"Model: {Path(model_name).name} | {str(original_type)[6:]} → int{q_bits}", case_dir / "heatmap_mae.png", case_dir / "heatmap_mae.pdf")

    df = DataFrame(global_heatmap)
    df = df.groupby(["quantized layer", "measured hidden"], as_index=False)[["cosine"]].mean()
    plot_heatmap(df, "quantized layer", "measured hidden", "cosine", "Measured hidden", "Quantized layer", f"Model: {Path(model_name).name} | {str(original_type)[6:]} → int{q_bits}", case_dir / "heatmap_cos_sim.png", case_dir / "heatmap_cos_sim.pdf")
