"""
Script to analyze the presence of the "attention shift" phenomenon.
"""
import matplotlib.pyplot as plt
import modal
import os

from uuid import uuid4

from utils.modal_utils import get_modal_image

app = modal.App(name="attention-shift")
image = get_modal_image()


@app.function(image=image, gpu="A100", timeout=80*60)
def run(
    model_name,
    dataset,
    split,
    num_samples,
    width,
    height,
):
    import torch
    from datasets import load_dataset

    from model import load_model, run_prefill

    device = torch.device("cuda")

    model, processor = load_model(model_name)

    dataset_iterator = iter(load_dataset(dataset, split=split, streaming=True))

    samples = []
    while len(samples) < num_samples:
        sample = next(dataset_iterator)
        if sample["image"].width != width or sample["image"].height != height:
            continue
        samples.append(sample)


    running_sums = None
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1} of {len(samples)}...")
        result = run_prefill(
            model,
            processor,
            sample["image"],
            sample["question"],
        )

        prefill_attn_scores = result.captured_attentions.mean(dim=2).squeeze(dim=1) # (L, 1, M, T, T) -> (L, T, T)
        last_token_attn_scores = prefill_attn_scores[:, -1, result.token_types == 1] # (L, T, T) -> (L, V)

        if running_sums is None:
            running_sums = torch.zeros_like(last_token_attn_scores).to(device)

        running_sums += (last_token_attn_scores - last_token_attn_scores.mean()) / last_token_attn_scores.std()

    running_sums /= num_samples

    return running_sums.cpu()

@app.local_entrypoint()
def main(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    dataset="merve/vqav2-small",
    split="validation",
    num_samples: int = 1,
    width: int = 640,
    height: int = 480,
):
    id = str(uuid4())
    print(f"Run UUID: {id}")

    averaged_attn_scores = run.remote(
        model_name=model_name,
        dataset=dataset,
        split=split,
        num_samples=num_samples,
        width=width,
        height=height,
    )

    print(f"Averaged Attention Scores Shape: {averaged_attn_scores.shape}")

    os.makedirs(f"assets/attention_shift/{id}/", exist_ok=True)

    V = averaged_attn_scores.shape[1]
    for layer in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]:
        layer_attn = averaged_attn_scores[layer]
        
        plt.figure(figsize=(5, 5))
        plt.bar(range(V), layer_attn.detach().numpy())
        plt.xlabel("Visual Token Index")
        plt.ylabel("Attention Score")
        plt.title(f"Layer {layer} Image Attention Distribution")
        plt.savefig(f"assets/attention_shift/{id}/layer_{layer}_last_token_attention_distribution.png")
        plt.close()





