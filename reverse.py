from gc import collect

from json import dump

from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig

from datasets import load_dataset

import matplotlib.pyplot as plt

import torch

from pandas import DataFrame

from pathlib import Path


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


def plot_heatmap(dataframe, index, columns, value, xlabel, ylabel, title, output_file):
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

    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


device = "cuda" if (p := torch.cuda.is_available()) else "cpu"

original_types = (torch.float16, torch.float32, torch.float64)

BITS = 4

model_names = ("HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct")

prompts = ["Explain gravity.", "What is 173 × 29?", "Write a Python function to reverse a list.", "Translate 'Good morning' into Bulgarian.", "Why is the sky blue?"]

START, END = 0, -1
experiments = []

for model_name in model_names:
    for original_type in original_types:
        experiments.append((model_name, original_type))

for case, (model_name, original_type) in enumerate(experiments[START:END], START):
    print(case, model_name, original_type)

    case_dir = Path(f"reverse_results/case{case:03d}")
    case_dir.mkdir(parents=True, exist_ok=True)

    quantized_name = model_name.split("/")[-1] + f"-GPTQ-{BITS}bit"

    with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
        dump({"Device": device, "Model": model_name, "Original type": str(original_type)[6:], "Quantized model": quantized_name, "quantization": f"GPTQ int{BITS}", "prompts": prompts}, f, indent=4)

    prompts_dir = case_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    global_fp_heatmap_mae, global_q_heatmap_mae = [], []

    global_fp_q_damage = {"mae": [], "cosine": []}

    # GPTQ calibration data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    calibration = [text for text in dataset["text"] if text.strip()][:128]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = GPTQConfig(bits=BITS, dataset=calibration, tokenizer=tokenizer)

    for i, prompt in enumerate(prompts):
        print(prompt)

        prompt_dir = prompts_dir / f"prompt{i}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        prompt_fp_heatmap_mae, prompt_q_heatmap_mae, prompt_result_json = [], [], []

        with open(prompt_dir / "content.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # ------------------------------------------------------------
        # Fully unquantized model
        # ------------------------------------------------------------

        unquantized = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)
        unquantized.eval()

        with torch.inference_mode():
            outputs_unquantized = unquantized(**inputs,  output_hidden_states=True)

            unquantized_hidden = outputs_unquantized.hidden_states

        # Save the original transformer layers so that individual
        # layers can later be restored in the GPTQ model.
        unquantized_layers = [{key: value.cpu().clone() for key, value in layer.state_dict().items()} for layer in unquantized.model.layers]

        del unquantized, outputs_unquantized

        # ------------------------------------------------------------
        # Fully GPTQ-quantized model
        # ------------------------------------------------------------

        quantized = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=config).to(device)

        quantized.eval()

        with torch.inference_mode():
            outputs_quantized = quantized(**inputs, output_hidden_states=True)

            quantized_hidden = outputs_quantized.hidden_states

        # ------------------------------------------------------------
        # Fully unquantized vs fully quantized
        # ------------------------------------------------------------

        prompt_fp_q_damage = {"layer": [], "mae": [], "cosine": []}

        global_fp_q_damage["mae"].append([])
        global_fp_q_damage["cosine"].append([])

        for k, (fp, q) in enumerate(zip(unquantized_hidden, quantized_hidden)):
            mae = torch.mean(torch.abs(fp.float() - q.float())).item()
            cosine = torch.nn.functional.cosine_similarity(fp.float().flatten(), q.float().flatten(), dim=0).item()

            prompt_fp_q_damage["layer"].append(k)
            prompt_fp_q_damage["mae"].append(mae)
            prompt_fp_q_damage["cosine"].append(cosine)

            global_fp_q_damage["mae"][i].append(mae)
            global_fp_q_damage["cosine"][i].append(cosine)

        plot_damage(
            prompt_fp_q_damage["layer"],
            prompt_fp_q_damage["mae"],
            prompt_fp_q_damage["cosine"],
            f"{Path(model_name).name} | Prompt {i}\nUnquantized vs GPTQ int{BITS}", prompt_dir,f"prompt{i}_unquantized_quantized_damage")

        del prompt_fp_q_damage

        # ------------------------------------------------------------
        # Restore one layer at a time
        # ------------------------------------------------------------

        for j, layer in enumerate(quantized.model.layers):
            fp_result_json, q_result_json = [], []

            spared_layer_dir = prompt_dir / f"spared_layer{j:03d}"
            spared_layer_dir.mkdir(parents=True, exist_ok=True)

            fp_damage_plot = {"layer": [], "mae": [], "cosine": []}
            q_damage_plot = {"layer": [], "mae": [], "cosine": []}

            with torch.no_grad():
                restore_layer = {key: value.cpu().clone() for key, value in layer.state_dict().items()}

            try:
                # Restore this layer to its original
                # floating-point weights.
                layer.load_state_dict(unquantized_layers[j])

                with torch.inference_mode():
                    outputs_hybrid = quantized(**inputs, output_hidden_states=True)

                for k, (fp, q, hybrid) in enumerate(zip(unquantized_hidden, quantized_hidden, outputs_hybrid.hidden_states)):
                    mae_fp = torch.mean(torch.abs(fp.float() - hybrid.float())).item()
                    mae_q = torch.mean(torch.abs(q.float() - hybrid.float())).item()

                    cosine_fp = torch.nn.functional.cosine_similarity(fp.float().flatten(), hybrid.float().flatten(), dim=0).item()
                    cosine_q = torch.nn.functional.cosine_similarity(q.float().flatten(), hybrid.float().flatten(), dim=0).item()

                    fp_damage_plot["layer"].append(k)
                    fp_damage_plot["mae"].append(mae_fp)
                    fp_damage_plot["cosine"].append(cosine_fp)

                    q_damage_plot["layer"].append(k)
                    q_damage_plot["mae"].append(mae_q)
                    q_damage_plot["cosine"].append(cosine_q)

                    prompt_fp_heatmap_mae.append({"spared_layer": j, "measured_layer": k, "mae": mae_fp})
                    global_fp_heatmap_mae.append({"prompt": i, "spared_layer": j, "measured_layer": k, "mae": mae_fp})

                    prompt_q_heatmap_mae.append({"spared_layer": j, "measured_layer": k, "mae": mae_q})
                    global_q_heatmap_mae.append({"prompt": i, "spared_layer": j, "measured_layer": k, "mae": mae_q})

                    fp_result_json.append({"measured layer": k, "mae": mae_fp, "cosine": cosine_fp})
                    q_result_json.append({"measured layer": k, "mae": mae_q, "cosine": cosine_q})

                    prompt_result_json.append({"spared layer": j, "measured layer": k, "mae vs unquantized": mae_fp, "cosine vs unquantized": cosine_fp, "mae vs quantized": mae_q, "cosine vs quantized": cosine_q})

                del outputs_hybrid

            finally:
                layer.load_state_dict(restore_layer)

                del restore_layer

            # --------------------------------------------------------
            # Save per-spared-layer JSON
            # --------------------------------------------------------

            with open(spared_layer_dir / "fp_results.json", "w", encoding="utf-8") as f:
                dump(fp_result_json, f, indent=4)

            with open(spared_layer_dir / "q_results.json", "w", encoding="utf-8") as f:
                dump(q_result_json, f, indent=4)

            # --------------------------------------------------------
            # Plot damage relative to fully unquantized model
            # --------------------------------------------------------

            plot_damage(fp_damage_plot["layer"], fp_damage_plot["mae"], fp_damage_plot["cosine"], f"{Path(model_name).name} | Prompt {i}\nSpared layer {j} | Hybrid vs unquantized", spared_layer_dir, "damage_vs_unquantized")

            # --------------------------------------------------------
            # Plot damage relative to fully quantized model
            # --------------------------------------------------------

            plot_damage(q_damage_plot["layer"], q_damage_plot["mae"], q_damage_plot["cosine"], f"{Path(model_name).name} | Prompt {i}\nSpared layer {j} | Hybrid vs GPTQ", spared_layer_dir, "damage_vs_quantized")

            del fp_damage_plot, q_damage_plot

        del quantized, unquantized_hidden, quantized_hidden

        collect()

        if p:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ------------------------------------------------------------
        # Save prompt-level results
        # ------------------------------------------------------------

        with open(prompt_dir / f"prompt{i}_results.json", "w", encoding="utf-8") as f:
            dump(prompt_result_json, f, indent=4)

        # ------------------------------------------------------------
        # Prompt heatmaps
        # ------------------------------------------------------------

        df = DataFrame(prompt_fp_heatmap_mae)
        plot_heatmap(df, "spared_layer", "measured_layer", "mae", "Measured layer", "Spared layer", f"{Path(model_name).name} | Prompt {i}\nHybrid vs unquantized | GPTQ int{BITS}", prompt_dir / "heatmap_fp_mae.png")

        df = DataFrame(prompt_q_heatmap_mae)
        plot_heatmap(df, "spared_layer", "measured_layer", "mae", "Measured layer", "Spared layer", f"{Path(model_name).name} | Prompt {i}\nHybrid vs GPTQ | GPTQ int{BITS}", prompt_dir / "heatmap_q_mae.png")

        del prompt_result_json

    # ------------------------------------------------------------
    # Global unquantized-vs-quantized damage
    # ------------------------------------------------------------

    n = len(global_fp_q_damage["mae"][0])

    global_fp_q_damage["layer"] = list(range(n))
    global_fp_q_damage["mae"] = [sum(prompt_results[layer] for prompt_results in global_fp_q_damage["mae"]) / len(global_fp_q_damage["mae"]) for layer in range(n)]
    global_fp_q_damage["cosine"] = [sum(prompt_results[layer] for prompt_results in global_fp_q_damage["cosine"]) / len(global_fp_q_damage["cosine"]) for layer in range(n)]

    plot_damage(global_fp_q_damage["layer"], global_fp_q_damage["mae"], global_fp_q_damage["cosine"], f"{Path(model_name).name}\nMean unquantized vs GPTQ int{BITS} across {len(prompts)} prompts", prompts_dir, "unquantized_quantized_damage")

    # ------------------------------------------------------------
    # Global heatmaps
    # ------------------------------------------------------------

    df = DataFrame(global_fp_heatmap_mae)
    df = df.groupby(["spared_layer", "measured_layer"], as_index=False)[["mae"]].mean()
    plot_heatmap(df, "spared_layer", "measured_layer", "mae", "Measured layer", "Spared layer", f"{Path(model_name).name}\nMean hybrid vs unquantized MAE across {len(prompts)} prompts", case_dir / "heatmap_fp_mae.png")

    df = DataFrame(global_q_heatmap_mae)
    df = df.groupby(["spared_layer", "measured_layer"], as_index=False)[["mae"]].mean()
    plot_heatmap(df, "spared_layer", "measured_layer", "mae", "Measured layer", "Spared layer", f"{Path(model_name).name}\nMean hybrid vs GPTQ MAE across {len(prompts)} prompts", case_dir / "heatmap_q_mae.png")

    del tokenizer, inputs

    collect()

    if p:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
