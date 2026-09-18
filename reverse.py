from gc import collect

from json import dump

from statistics import mean

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


def plot_damage(layers, maes, cosines, title, output_dir, filename):
    fig, ax1 = plt.subplots(figsize=(9, 4))

    ax1.plot(layers, maes, marker="o", color="tab:red")
    ax1.set_xlabel("Measured hidden")
    ax1.set_ylabel("MAE", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()

    ax2.plot(layers, cosines, marker="s", color="tab:blue")
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

BITS = 4

model_names = ("HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B-Instruct", "allenai/OLMo-1B-hf",
               "allenai/OLMo-7B-hf", "microsoft/Phi-3-mini-4k-instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct")

prompts = ["Explain gravity.", "What is 173 × 29?", "Write a Python function to reverse a list.", "Translate 'Good morning' into Bulgarian.", "Why is the sky blue?"]

START = 0
experiments = []

for model_name in model_names:
    for original_type in (torch.float16, torch.float32, torch.float64):
        experiments.append((model_name, original_type))

for case, (model_name, original_type) in enumerate(experiments[START:], START):
    print(case, model_name, original_type)

    case_dir = Path(f"reverse/case{case:03d}")
    case_dir.mkdir(parents=True, exist_ok=True)

    case_fp_h_dir = case_dir / "unquantized_vs_hybrid"
    case_fp_h_dir.mkdir(parents=True, exist_ok=True)

    with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
        dump({"Device": device, "Model": model_name, "Original type": str(original_type)[6:], "quantization": f"int{BITS}", "prompts": prompts}, f, indent=4)

    prompts_dir = case_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    global_fp_h_heatmap = []
    global_fp_q_damage = {"mae": [], "cosine": []}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, prompt in enumerate(prompts):
        print(prompt)

        prompt_dir = prompts_dir / f"prompt{i}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        prompt_fp_h_dir = prompt_dir / "unquantized_vs_hybrid"
        prompt_fp_h_dir.mkdir(parents=True, exist_ok=True)

        prompt_fp_h_heatmap, prompt_result_json = [], []

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

        prompt_fp_q_damage = []

        global_fp_q_damage["mae"].append([])
        global_fp_q_damage["cosine"].append([])

        for k, (fp, q) in enumerate(list(zip(unquantized_hidden, quantized_hidden))[1:], 1):
            mae = torch.mean(torch.abs(fp.float() - q.float())).item()
            cosine = torch.nn.functional.cosine_similarity(fp.double().flatten(), q.double().flatten(), dim=0).item()

            prompt_fp_q_damage.append({"layer": k, "mae": mae, "cosine": cosine})

            global_fp_q_damage["mae"][i].append(mae)
            global_fp_q_damage["cosine"][i].append(cosine)

        with open(prompt_dir / f"prompt{i}_fp_q_damage.json", "w", encoding="utf-8") as f:
            dump(prompt_fp_q_damage, f, indent=4)

        plot_damage([curr["layer"] for curr in prompt_fp_q_damage], [curr["mae"] for curr in prompt_fp_q_damage], [curr["cosine"] for curr in prompt_fp_q_damage], f"{Path(model_name).name} | Prompt {i}\nUnquantized vs quantized int{BITS}", prompt_dir,f"prompt{i}_fp_q_damage")

        del prompt_fp_q_damage

        for j, layer in enumerate(quantized.model.layers):
            fp_h_result_json = []

            s_layer = prompt_dir / f"spared_layer{j:03d}"
            s_layer.mkdir(parents=True, exist_ok=True)

            fp_h_damage_plot = {"layer": [], "mae": [], "cosine": []}

            with torch.no_grad():
                restore_layer = {key: value.cpu().clone() for key, value in layer.state_dict().items()}

            try:
                layer.load_state_dict(unquantized_layers[j])

                with torch.inference_mode():
                    outputs_hybrid = quantized(**inputs, output_hidden_states=True)

                for k, (fp, q, hybrid) in enumerate(list(zip(unquantized_hidden, quantized_hidden, outputs_hybrid.hidden_states))[1:], 1):
                    mae = torch.mean(torch.abs(fp.float() - hybrid.float())).item()
                    cosine = torch.nn.functional.cosine_similarity(fp.double().flatten(), hybrid.double().flatten(), dim=0).item()

                    fp_h_damage_plot["layer"].append(k)
                    fp_h_damage_plot["mae"].append(mae)
                    fp_h_damage_plot["cosine"].append(cosine)

                    prompt_fp_h_heatmap.append({"spared layer": j, "measured hidden": k, "mae": mae, "cosine": cosine})
                    global_fp_h_heatmap.append({"prompt": prompt, "spared layer": j, "measured hidden": k, "mae": mae, "cosine": cosine})

                    fp_h_result_json.append({"measured hidden": k, "mae": mae, "cosine": cosine})
                    prompt_result_json.append({"spared layer": j, "measured hidden": k, "mae": mae, "cosine": cosine})

                with open(s_layer / f"layer{j}_unquantized_vs_hybrid.json", "w", encoding="utf-8") as f:
                    dump(fp_h_result_json, f, indent=4)

                del outputs_hybrid

            finally:
                layer.load_state_dict(restore_layer)

                del restore_layer

            plot_damage(fp_h_damage_plot["layer"], fp_h_damage_plot["mae"], fp_h_damage_plot["cosine"], f"{Path(model_name).name} | Prompt {i}\nSpared layer {j} | Unquantized vs hybrid", s_layer, f"layer{j}_damage")

            del fp_h_damage_plot

        del quantized, unquantized_hidden, quantized_hidden

        collect()

        if p:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with open(prompt_fp_h_dir / f"prompt{i}_results.json", "w", encoding="utf-8") as f:
            dump(prompt_result_json, f, indent=4)

        df = DataFrame(prompt_fp_h_heatmap)
        plot_heatmap(df, "spared layer", "measured hidden", "mae", "Measured hidden", "Spared layer", f"{Path(model_name).name} | Prompt {i}\nUnquantized vs hybrid MAE | int{BITS}", prompt_fp_h_dir / "heatmap_mae.png", prompt_fp_h_dir / "heatmap_mae.pdf")

        df = DataFrame(prompt_fp_h_heatmap)
        plot_heatmap(df, "spared layer", "measured hidden", "cosine", "Measured hidden", "Spared layer", f"{Path(model_name).name} | Prompt {i}\nUnquantized vs hybrid cos sim | int{BITS}", prompt_fp_h_dir / "heatmap_cos_sim.png", prompt_fp_h_dir / "heatmap_cos_sim.pdf")

        del prompt_result_json

    n = len(global_fp_q_damage["mae"][0])

    global_fp_q_damage["layer"] = list(range(1, n + 1))
    global_fp_q_damage["mae"] = [mean(prompt_results[layer] for prompt_results in global_fp_q_damage["mae"]) for layer in range(n)]
    global_fp_q_damage["cosine"] = [mean(prompt_results[layer] for prompt_results in global_fp_q_damage["cosine"]) for layer in range(n)]

    plot_damage(global_fp_q_damage["layer"], global_fp_q_damage["mae"], global_fp_q_damage["cosine"], f"{Path(model_name).name}\nMean unquantized vs quantized int{BITS} across {len(prompts)} prompts", case_dir, "fp_q_damage")

    df = DataFrame(global_fp_h_heatmap)
    df = df.groupby(["spared layer", "measured hidden"], as_index=False)[["mae"]].mean()
    plot_heatmap(df, "spared layer", "measured hidden", "mae", "Measured hidden", "Spared layer", f"{Path(model_name).name}\nMean unquantized vs hybrid MAE across {len(prompts)} prompts", case_fp_h_dir / "heatmap_mae.png", case_fp_h_dir / "heatmap_mae.pdf")

    df = DataFrame(global_fp_h_heatmap)
    df = df.groupby(["spared layer", "measured hidden"], as_index=False)[["cosine"]].mean()
    plot_heatmap(df, "spared layer", "measured hidden", "cosine", "Measured hidden", "Spared layer", f"{Path(model_name).name}\nMean unquantized vs hybrid cos sim across {len(prompts)} prompts", case_fp_h_dir / "heatmap_cos_sim.png", case_fp_h_dir / "heatmap_cos_sim.pdf")

    del tokenizer, inputs

    collect()

    if p:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
