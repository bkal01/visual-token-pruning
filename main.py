import os
import modal

from uuid import uuid4

from visualization import plot_attn_heatmap, plot_last_attn_intensity_distribution

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

@app.function(image=image, gpu="A100")
def run_vlm():
    from datasets import load_dataset

    from model import load_model, run_inference

    model, processor = load_model("Qwen/Qwen3-VL-2B-Instruct")
    vqav2_small = load_dataset("merve/vqav2-small", split="validation", streaming=True)
    sample = next(iter(vqav2_small))

    print(f"Image: {sample['image']}")
    print(f"Question: {sample['question']}")
    input_ids, generated_ids, prefill_attn_scores = run_inference(
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
    return output_text, prefill_attn_scores


@app.local_entrypoint()
def main():
    id = str(uuid4())
    print(f"Run UUID: {id}")
    os.makedirs(f"assets/runs/{id}/", exist_ok=True)


    output_text,prefill_attn_scores = run_vlm.remote()
    print(f"Model Output: {output_text}")


    layer_1_scores, layer_14_scores, layer_28_scores = prefill_attn_scores[0], prefill_attn_scores[13], prefill_attn_scores[27]
    
    # average across heads
    layer_1_scores, layer_14_scores, layer_28_scores = layer_1_scores.mean(dim=1).squeeze(dim=0), layer_14_scores.mean(dim=1).squeeze(dim=0), layer_28_scores.mean(dim=1).squeeze(dim=0)
    plot_attn_heatmap(layer_1_scores, save_path=f"assets/runs/{id}/layer_1_attention_heatmap.png")
    plot_attn_heatmap(layer_14_scores, save_path=f"assets/runs/{id}/layer_14_attention_heatmap.png")
    plot_attn_heatmap(layer_28_scores, save_path=f"assets/runs/{id}/layer_28_attention_heatmap.png")

    plot_last_attn_intensity_distribution(layer_1_scores, save_path=f"assets/runs/{id}/layer_1_last_attention_distribution.png")
    plot_last_attn_intensity_distribution(layer_14_scores, save_path=f"assets/runs/{id}/layer_14_last_attention_distribution.png")
    plot_last_attn_intensity_distribution(layer_28_scores, save_path=f"assets/runs/{id}/layer_28_last_attention_distribution.png")
    
