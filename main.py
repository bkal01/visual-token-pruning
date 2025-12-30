import os
import matplotlib.pyplot as plt
import modal

from uuid import uuid4

from visualization import (
    plot_attn_heatmap,
    plot_last_attn_intensity_distribution,
    plot_attn_visual_tokens_distribution,
)

app = modal.App(name="visual-token-pruning")

image = (
    modal.Image.debian_slim(
        python_version="3.10",
    )
    .apt_install(
        "git",
    )
    .uv_pip_install(
        [
            "transformers>=4.57.0",
            "torch>=2.6.0",
            "torchvision>=0.15.0",
            "pillow>=10.0.0",
            "matplotlib",
            "datasets",
        ],
    )
    .add_local_dir(
        local_path="models",
        remote_path="/root/models",
    )
    .add_local_file(
        local_path="visualization.py",
        remote_path="/root/visualization.py",
    )
    .add_local_file(
        local_path="model.py",
        remote_path="/root/model.py",
    )
)

@app.function(image=image, gpu="A100", timeout=600)
def run_vlm(num_samples, use_standard_rope):
    import torch
    from datasets import load_dataset

    from model import load_model, run_inference

    device = torch.device("cuda")

    model, processor = load_model("Qwen/Qwen3-VL-2B-Instruct")
    vqav2_small = iter(load_dataset("merve/vqav2-small", split="validation", streaming=True))

    # maintain a running sum of the attention scores for each layer
    # magic numbers! 28 is number of layers, 260 is number of visual tokens for images with width 640 in VQAv2
    running_sums = torch.zeros((28, 260), device=device)
    token_types_list = []
    processed_samples = 0

    while processed_samples < num_samples:
        print(f"Processing sample {processed_samples+1} of {num_samples}...")
        sample = next(vqav2_small)
        print(f"Sample resolution: {sample['image'].width}x{sample['image'].height}")
        # we control for image width & height here to ensure # visual tokens is the same for all samples in VQAv2
        if sample["image"].width != 640 or sample["image"].height != 424:
            print(f"Skipping sample {processed_samples+1}: image width is {sample['image'].width}, not 640")
            continue
        print(f"Question: {sample['question']}")
        input_ids, token_types, generated_ids, prefill_attn_scores = run_inference(
            model,
            processor,
            sample["image"],
            sample["question"],
            max_new_tokens=128,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"\n\nOutput Text: {output_text}\n\n")

        print(f"Prefill Attn Scores Shape: {prefill_attn_scores.shape}")
        prefill_attn_scores = prefill_attn_scores.mean(dim=2).squeeze(dim=1) # (L, 1, M, T, T) -> (L, T, T)
        prefill_attn_scores = prefill_attn_scores[:, :, token_types == 1] # (L, T, T) -> (L, T, V)
        print(f"Prefill Attn Scores after visual token mask: {prefill_attn_scores.shape}")

        last_token_attn_scores = prefill_attn_scores[:, -1, :] # (L, T, V) -> (L, V)

        running_sums += last_token_attn_scores
        token_types_list.append(token_types.cpu())
        if processed_samples > 0:
            if not torch.sum(token_types_list[-1]) == torch.sum(token_types_list[-2]):
                print(f"Number of visual tokens changed from {torch.sum(token_types_list[-2])} to {torch.sum(token_types_list[-1])}")
                break
        processed_samples += 1
    
    running_sums /= num_samples


    return running_sums.cpu(), token_types_list[0].cpu()


@app.local_entrypoint()
def main(num_samples=5, use_standard_rope=False):
    id = str(uuid4())
    print(f"Run UUID: {id}")
    os.makedirs(f"assets/runs/{id}/", exist_ok=True)


    averaged_attn_scores, token_types = run_vlm.remote(
        num_samples=int(num_samples),
        use_standard_rope=bool(use_standard_rope),
    )


    layer_indices = [0, 3, 6, 9, 12, 15, 18, 21, 24]  # layers 1, 4, 7, 10, 13, 16, 19, 22, 25
    V = averaged_attn_scores.shape[1]
    
    _, axes = plt.subplots(3, 3, figsize=(12, 9))
    
    for i, layer_idx in enumerate(layer_indices):
        row, col = i // 3, i % 3
        scores = averaged_attn_scores[layer_idx].cpu().numpy()
        axes[row, col].bar(range(V), scores)
        axes[row, col].set_xlabel("Visual Token Index")
        axes[row, col].set_ylabel("Attention Score")
        axes[row, col].set_title(f"Layer {layer_idx + 1}")
    
    plt.tight_layout()
    plt.show()


    
