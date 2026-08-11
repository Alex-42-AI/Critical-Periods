from gc import collect

from json import dump

from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt

import torch

from pandas import DataFrame

from pathlib import Path


def quantize_tensor(weight):
    max_int = 2 ** (BITS - 1) - 1
    scale = weight.abs().max() / max_int

    return torch.round(weight / scale) * scale


def quantize(lyr):
    with torch.no_grad():
        for module in lyr.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight
                quantized = quantize_tensor(weight)
                weight.copy_(quantized)


def plot_damage(layers, mae, cosine, title, output_dir, filename):
    fig, ax1 = plt.subplots(figsize=(9, 4))

    ax1.plot(layers, mae, marker="o", color="tab:red")
    ax1.set_xlabel("Measured layer")
    ax1.set_ylabel("MAE", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()

    ax2.plot(layers, cosine, marker="s", color="tab:blue")
    ax2.set_ylabel("Cosine similarity", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title(title)

    ax1.grid(True)
    fig.tight_layout()

    plt.savefig(output_dir / f"{filename}.png", bbox_inches="tight")
    plt.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")

    plt.close(fig)


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

BITS = 4

model_names = ("HuggingFaceTB/SmolLM3-3B", "mistralai/Mistral-7B-Instruct-v0.3", "allenai/OLMo-1B-hf", "allenai/OLMo-7B-hf", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B")[2:]

prompts = ["Explain gravity.", "What is 173 × 29?", "Write a Python function to reverse a list.", "Translate 'Good morning' into Bulgarian.", "Why is the sky blue?"]

START = 0
experiments = []

for model_name in model_names:
    for original_type in original_types:
        experiments.append((model_name, original_type))

for case, (model_name, original_type) in enumerate(experiments[START:], START + 15):
    print(case, model_name, original_type)

    case_dir = Path(f"reverse_new/case{case:03d}")
    case_dir.mkdir(parents=True, exist_ok=True)

    case_unquantized_dir = case_dir / "unquantized"
    case_unquantized_dir.mkdir(parents=True, exist_ok=True)

    with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
        dump({"Device": device, "Model": model_name, "Original type": str(original_type)[6:], "quantization": f"int{BITS}", "prompts": prompts}, f, indent=4)

    prompts_dir = case_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    global_fp_heatmap_mae, global_fp_RMSNorm_json = [], []
    global_fp_q_RMSNorm_json = []
    global_fp_q_damage = {"mae": [], "cosine": []}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, prompt in enumerate(prompts):
        print(prompt)

        prompt_dir = prompts_dir / f"prompt{i}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        prompt_unquantized_dir = prompt_dir / "unquantized"
        prompt_unquantized_dir.mkdir(parents=True, exist_ok=True)

        prompt_fp_heatmap_mae, prompt_result_json = [], []
        prompt_fp_RMSNorm_json = []

        with open(prompt_dir / "content.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        unquantized = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)
        unquantized.eval()

        with torch.inference_mode():
            outputs_unquantized = unquantized(**inputs,  output_hidden_states=True)
            unquantized_hidden = outputs_unquantized.hidden_states

        unquantized_layers = [{key: value.cpu().clone() for key, value in layer.state_dict().items()} for layer in unquantized.model.layers]

        del unquantized, outputs_unquantized

        quantized = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)

        for layer in quantized.model.layers:
            quantize(layer)

        quantized.eval()

        with torch.inference_mode():
            outputs_quantized = quantized(**inputs, output_hidden_states=True)
            quantized_hidden = outputs_quantized.hidden_states

        prompt_fp_q_damage = {"layer": [], "mae": [], "cosine": []}

        global_fp_q_damage["mae"].append([])
        global_fp_q_damage["cosine"].append([])

        for k, (fp, q) in enumerate(list(zip(unquantized_hidden, quantized_hidden))[:-1]):
            mae = torch.mean(torch.abs(fp.float() - q.float())).item()
            cosine = torch.nn.functional.cosine_similarity(fp.float().flatten(), q.float().flatten(), dim=0).item()

            prompt_fp_q_damage["layer"].append(k)
            prompt_fp_q_damage["mae"].append(mae)
            prompt_fp_q_damage["cosine"].append(cosine)

            global_fp_q_damage["mae"][i].append(mae)
            global_fp_q_damage["cosine"][i].append(cosine)

        plot_damage(prompt_fp_q_damage["layer"], prompt_fp_q_damage["mae"], prompt_fp_q_damage["cosine"], f"{Path(model_name).name} | Prompt {i}\nUnquantized vs quantized int{BITS}", prompt_dir,f"prompt{i}_fp_q_damage")

        del prompt_fp_q_damage

        with open(prompt_dir / "fp_q_RMSNorm.json", "w") as f:
            fp, q = unquantized_hidden[-1], quantized_hidden[-1]

            mae = torch.mean(torch.abs(fp.float() - q.float())).item()
            cosine = torch.nn.functional.cosine_similarity(fp.float().flatten(), q.float().flatten(), dim=0).item()

            global_fp_q_RMSNorm_json.append({"prompt": prompt, "mae": mae, "cos": cosine})
            dump({"mae": mae, "cos": cosine}, f, indent=4)

        for j, layer in enumerate(quantized.model.layers):
            fp_result_json = []

            s_layer = prompt_dir / f"spared_layer{j:03d}"
            s_layer.mkdir(parents=True, exist_ok=True)

            layer_unquantized_dir = s_layer / "unquantized"
            layer_unquantized_dir.mkdir(parents=True, exist_ok=True)

            fp_damage_plot = {"layer": [], "mae": [], "cosine": []}

            with torch.no_grad():
                restore_layer = {key: value.cpu().clone() for key, value in layer.state_dict().items()}

            try:
                layer.load_state_dict(unquantized_layers[j])

                with torch.inference_mode():
                    outputs_hybrid = quantized(**inputs, output_hidden_states=True)

                for k, (fp, q, hybrid) in enumerate(list(zip(unquantized_hidden, quantized_hidden, outputs_hybrid.hidden_states))[:-1]):
                    mae_fp = torch.mean(torch.abs(fp.float() - hybrid.float())).item()

                    cosine_fp = torch.nn.functional.cosine_similarity(fp.float().flatten(), hybrid.float().flatten(), dim=0).item()

                    fp_damage_plot["layer"].append(k)
                    fp_damage_plot["mae"].append(mae_fp)
                    fp_damage_plot["cosine"].append(cosine_fp)

                    prompt_fp_heatmap_mae.append({"spared_layer": j, "measured_layer": k, "mae": mae_fp})
                    global_fp_heatmap_mae.append({"prompt": i, "spared_layer": j, "measured_layer": k, "mae": mae_fp})

                    fp_result_json.append({"measured layer": k, "mae": mae_fp, "cosine": cosine_fp})

                    prompt_result_json.append({"spared layer": j, "measured layer": k, "mae vs unquantized": mae_fp, "cosine vs unquantized": cosine_fp})

                with open(layer_unquantized_dir / f"layer{j}_damage_results.json", "w", encoding="utf-8") as f:
                    dump(fp_result_json, f, indent=4)

                with open(layer_unquantized_dir / f"layer{j}RMSNorm.json", "w") as f:
                    fp, hybrid = unquantized_hidden[-1], outputs_hybrid.hidden_states[-1]

                    mae = torch.mean(torch.abs(fp.float() - hybrid.float())).item()
                    cos = torch.nn.functional.cosine_similarity(fp.float().flatten(), hybrid.float().flatten(), 0).item()

                    dump({"mae": mae, "cos": cos}, f, indent=4)
                    prompt_fp_RMSNorm_json.append({"quantized layer": j, "mae": mae, "cos": cos})
                    global_fp_RMSNorm_json.append({"prompt": prompt, "quantized layer": j, "mae": mae, "cos": cos})

                del outputs_hybrid

            finally:
                layer.load_state_dict(restore_layer)

                del restore_layer

            plot_damage(fp_damage_plot["layer"], fp_damage_plot["mae"], fp_damage_plot["cosine"], f"{Path(model_name).name} | Prompt {i}\nSpared layer {j} | Hybrid vs unquantized", layer_unquantized_dir, f"layer{j}_damage")

            del fp_damage_plot

        del quantized, unquantized_hidden, quantized_hidden

        collect()

        if p:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with open(prompt_dir / f"prompt{i}_fp_q_results.json", "w", encoding="utf-8") as f:
            dump(prompt_result_json, f, indent=4)

        df = DataFrame(prompt_fp_heatmap_mae)
        plot_heatmap(df, "spared_layer", "measured_layer", "mae", "Measured layer", "Spared layer", f"{Path(model_name).name} | Prompt {i}\nHybrid vs unquantized | int{BITS}", prompt_unquantized_dir / "heatmap_mae.png", prompt_unquantized_dir / "heatmap_mae.pdf")

        with open(prompt_unquantized_dir / f"prompt{i}_RMSNorm.json", "w", encoding="utf-8") as f:
            dump(prompt_fp_RMSNorm_json, f, indent=4)

        del prompt_result_json

    with open(case_dir / "fp_q_RMSNorm.json", "w", encoding="utf-8") as f:
        dump(global_fp_q_RMSNorm_json, f, indent=4)

    n = len(global_fp_q_damage["mae"][0])

    global_fp_q_damage["layer"] = list(range(n))
    global_fp_q_damage["mae"] = [sum(prompt_results[layer] for prompt_results in global_fp_q_damage["mae"]) / len(global_fp_q_damage["mae"]) for layer in range(n)]
    global_fp_q_damage["cosine"] = [sum(prompt_results[layer] for prompt_results in global_fp_q_damage["cosine"]) / len(global_fp_q_damage["cosine"]) for layer in range(n)]

    plot_damage(global_fp_q_damage["layer"], global_fp_q_damage["mae"], global_fp_q_damage["cosine"], f"{Path(model_name).name}\nMean unquantized vs quantized int{BITS} across {len(prompts)} prompts", case_dir, "fp_q_damage")

    df = DataFrame(global_fp_heatmap_mae)
    df = df.groupby(["spared_layer", "measured_layer"], as_index=False)[["mae"]].mean()
    plot_heatmap(df, "spared_layer", "measured_layer", "mae", "Measured layer", "Spared layer", f"{Path(model_name).name}\nMean hybrid vs unquantized MAE across {len(prompts)} prompts", case_unquantized_dir / "heatmap_mae.png", case_unquantized_dir / "heatmap_mae.pdf")

    with open(case_unquantized_dir / "RMSNorm.json", "w", encoding="utf-8") as f:
        dump(global_fp_RMSNorm_json, f, indent=4)

    del tokenizer, inputs

    collect()

    if p:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
