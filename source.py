from gc import collect

from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt

import torch

from pandas import DataFrame

from pathlib import Path

from json import dump


def quantize_tensor(weight):
    max_int = 2 ** (q_bits - 1) - 1
    scale = weight.abs().max() / max_int

    return torch.round(weight / scale) * scale


def quantize(lyr):
    with torch.no_grad():
        for module in lyr.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight
                quantized = quantize_tensor(weight)
                weight.copy_(quantized)


# def last_layer_hook(module, inp, out):
#     captures["last_layer"] = out[0].detach()
#
#
# def norm_hook(module, inp, out):
#     captures["norm"] = out.detach()


device = "cuda" if (p := torch.cuda.is_available()) else "cpu"

original_types = (torch.float16, torch.float32, torch.float64)

q_bits_ls = (32, 16, 8, 4, 2)

model_names = ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM3-3B",
               "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3-mini-4k-instruct",
               "mistralai/Mistral-7B-Instruct-v0.3")

prompts = ["Explain gravity.", "What is 173 × 29?", "Write a Python function to reverse a list.",
           "Translate 'Good morning' into Bulgarian.", "Why is the sky blue?"]

case = 0
experiments = []

for model_name in model_names:
    for i, original_type in enumerate(original_types):
        for q_bits in q_bits_ls[2 - i:]:
            experiments.append((model_name, original_type, q_bits))

for model_name, original_type, q_bits in experiments:
    print(model_name, original_type, q_bits)

    case_dir = Path(f"results/case{case:03d}")
    case_dir.mkdir(parents=True)
    case += 1

    metadata_file = case_dir / "metadata.txt"

    prompts_dir = case_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "w") as f:
        f.write(f"Device: {device}\nModel: {model_name}\nOriginal type: {original_type}\nQuantization: int{q_bits}\n")

    global_heatmap_mae, global_result_json, global_RMSNorm_json = [], [], []

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, prompt in enumerate(prompts):
        print(prompt)

        prompt_dir = prompts_dir / f"prompt{i}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_heatmap_mae, prompt_result_json, prompt_RMSNorm_json = [], [], []

        with open(prompt_dir / "content.txt", "w", encoding="utf-8") as f:
            f.write(f"{prompt}\n")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)
        model.eval()

        # Verification experiment:
        # hidden_states[-1] corresponds to the output of the model's final RMSNorm,
        # rather than an additional transformer layer. This was verified by comparing
        # the final transformer-layer and RMSNorm outputs using forward hooks.
        # with torch.inference_mode():
        #     outputs = model(**inputs, output_hidden_states=True)
        #
        # captures = {}
        #
        # h1 = model.model.layers[-1].register_forward_hook(last_layer_hook)
        # h2 = model.model.norm.register_forward_hook(norm_hook)
        #
        # with torch.inference_mode():
        #     outputs = model(**inputs, output_hidden_states=True)
        #
        # h1.remove(), h2.remove()
        #
        # print(torch.mean(torch.abs(captures["last_layer"] - outputs.hidden_states[-1])))
        # print(torch.mean(torch.abs(captures["norm"] - outputs.hidden_states[-1])))
        # print(torch.mean(torch.abs(captures["last_layer"] - captures["norm"])))

        with torch.inference_mode():
            outputs_fp = model(**inputs, output_hidden_states=True)
            baseline_hidden = outputs_fp.hidden_states

        del model, outputs_fp

        model_q = AutoModelForCausalLM.from_pretrained(model_name, dtype=original_type).to(device)
        model_q.eval()

        for j, layer in enumerate(model_q.model.layers[:-1]):
            layer_result_json = []
            q_layer = prompt_dir / f"q_layer{j:03d}"
            q_layer.mkdir(exist_ok=True)
            damage_plot = {"layer": [], "mae": [], "cosine": []}

            with torch.no_grad():
                restore_layer = {k: v.clone() for k, v in layer.state_dict().items()}

            try:
                quantize(layer)

                with torch.inference_mode():
                    outputs_q = model_q(**inputs, output_hidden_states=True)

                for k, (fp, q) in enumerate(list(zip(baseline_hidden, outputs_q.hidden_states))[:-1]):
                    mae = torch.mean(torch.abs(fp.float() - q.float())).item()
                    cos = torch.nn.functional.cosine_similarity(fp.float().flatten(), q.float().flatten(), 0).item()

                    damage_plot["layer"].append(k), damage_plot["mae"].append(mae), damage_plot["cosine"].append(cos)

                    prompt_heatmap_mae.append({"quantized_layer": j, "measured_layer": k, "mae": mae})
                    global_heatmap_mae.append({"prompt": i, "quantized_layer": j, "measured_layer": k, "mae": mae})

                    layer_result_json.append({"measured layer": k, "mae": mae, "cos": cos})
                    prompt_result_json.append({"quantized layer": j, "measured layer": k, "mae": mae, "cos": cos})
                    global_result_json.append({"prompt": prompt, "quantized layer": j, "measured layer": k, "mae": mae, "cos": cos})

                with open(q_layer / f"layer{j}_RMSNorm.json", "w") as f:
                    fp, q = baseline_hidden[-1], outputs_q.hidden_states[-1]

                    mae = torch.mean(torch.abs(fp.float() - q.float())).item()
                    cos = torch.nn.functional.cosine_similarity(fp.float().flatten(), q.float().flatten(), 0).item()

                    dump({"mae": mae, "cos": cos}, f)
                    prompt_RMSNorm_json.append({"quantized layer": j, "mae": mae, "cos": cos})
                    global_RMSNorm_json.append({"prompt": prompt, "quantized layer": j, "mae": mae, "cos": cos})

                with open(q_layer / f"layer{j}_results.json", "w") as f:
                    dump(layer_result_json, f)

                    del layer_result_json

            finally:
                layer.load_state_dict(restore_layer)

                del restore_layer, outputs_q

            fig, ax1 = plt.subplots(figsize=(9, 4))

            ax1.plot(damage_plot["layer"], damage_plot["mae"], marker="o", color="tab:red")
            ax1.set_xlabel("Measured layer")
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

        del model_q, baseline_hidden, inputs

        collect()

        if p:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with open(prompt_dir / f"prompt{i}_results.json", "w") as f:
            dump(prompt_result_json, f)

            del prompt_result_json

        with open(prompt_dir / f"prompt{i}_RMSNorm.json", "w") as f:
            dump(prompt_RMSNorm_json, f)

            del prompt_RMSNorm_json

        df = DataFrame(prompt_heatmap_mae)

        pivot = df.pivot(index="quantized_layer", columns="measured_layer", values="mae")

        plt.figure(figsize=(10, 8))
        plt.imshow(pivot, aspect="auto", cmap="viridis")
        plt.title(f"{Path(model_name).name}\nPrompt {i}: MAE representation damage | {str(original_type)[6:]} → int{q_bits}")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel("Measured layer")
        plt.ylabel("Quantized layer")
        plt.colorbar(label="Hidden-state MAE")
        plt.savefig(prompt_dir / f"prompt{i}_heatmap_mae.png", bbox_inches="tight")
        plt.savefig(prompt_dir / f"prompt{i}_heatmap_mae.pdf", bbox_inches="tight")

    del tokenizer

    with open(case_dir / "global_results.json", "w") as f:
        dump(global_result_json, f)

        del global_result_json

    with open(case_dir / "global_RMSNorm.json", "w") as f:
        dump(global_RMSNorm_json, f)

        del global_RMSNorm_json

    df = DataFrame(global_heatmap_mae)
    df = (df.groupby(["quantized_layer", "measured_layer"], as_index=False)[["mae"]].mean())

    pivot = df.pivot(index="quantized_layer", columns="measured_layer", values="mae")

    plt.figure(figsize=(10, 8))
    plt.imshow(pivot, aspect="auto")
    plt.title(f"Model: {Path(model_name).name} | {str(original_type)[6:]} → int{q_bits}")
    plt.xlabel("Measured layer")
    plt.ylabel("Quantized layer")
    plt.colorbar(label="Hidden-state MAE")
    plt.savefig(case_dir / "heatmap_mae.png", bbox_inches="tight")
    plt.savefig(case_dir / "heatmap_mae.pdf", bbox_inches="tight")
