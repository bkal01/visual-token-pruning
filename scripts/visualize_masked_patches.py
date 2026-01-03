import modal
import numpy as np
import matplotlib.pyplot as plt

from utils.modal_utils import get_modal_image

app = modal.App(name="visualize-masked-patches")
image = get_modal_image()


@app.function(image=image, gpu="A100", timeout=10*60)
def run():
    import torch
    from datasets import load_dataset

    from model import load_model, run_inference
    from pruners.fastv_pruner import FastVPruner

    pruner = FastVPruner(
        layer_threshold=0,
        filtering_ratio=0.1,
    )
    model, processor = load_model(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        pruner=pruner,
    )

    dataset = load_dataset("merve/vqav2-small", split="validation", streaming=True)
    sample = next(iter(dataset))

    result = run_inference(
        model,
        processor,
        sample["image"],
        sample["question"],
    )

    initial_visual_count = len(result.inference_context.surviving_visual_indices[0])
    final_surviving = result.inference_context.surviving_visual_indices[-1].numpy()

    return {
        "image": sample["image"],
        "question": sample["question"],
        "initial_visual_count": initial_visual_count,
        "final_surviving_indices": final_surviving,
    }


def create_masked_image(pil_image, kept_mask, grid_h, grid_w):
    img_array = np.array(pil_image)
    h, w = img_array.shape[:2]
    
    patch_h = h / grid_h
    patch_w = w / grid_w
    
    masked_img = img_array.copy().astype(np.float32)
    
    for i in range(grid_h):
        for j in range(grid_w):
            idx = i * grid_w + j
            if not kept_mask[idx]:
                y_start = int(i * patch_h)
                y_end = int((i + 1) * patch_h)
                x_start = int(j * patch_w)
                x_end = int((j + 1) * patch_w)
                masked_img[y_start:y_end, x_start:x_end] = masked_img[y_start:y_end, x_start:x_end] * 0.2
    
    return masked_img.astype(np.uint8)


@app.local_entrypoint()
def main():
    data = run.remote()
    
    pil_image = data["image"]
    question = data["question"]
    initial_visual_count = data["initial_visual_count"]
    final_surviving = data["final_surviving_indices"]
    
    kept_mask = np.zeros(initial_visual_count, dtype=bool)
    kept_mask[final_surviving] = True
    
    grid_size = int(np.sqrt(initial_visual_count))
    if grid_size * grid_size != initial_visual_count:
        for gh in range(1, initial_visual_count + 1):
            if initial_visual_count % gh == 0:
                gw = initial_visual_count // gh
                if abs(gh - gw) < abs(grid_size - initial_visual_count // grid_size):
                    grid_size = gh
        grid_h, grid_w = grid_size, initial_visual_count // grid_size
    else:
        grid_h = grid_w = grid_size
    
    print(f"Visual tokens: {initial_visual_count}, Grid: {grid_h}x{grid_w}")
    print(f"Kept tokens: {len(final_surviving)}, Pruned: {initial_visual_count - len(final_surviving)}")
    
    masked_img = create_masked_image(pil_image, kept_mask, grid_h, grid_w)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(pil_image)
    axes[0].set_title("Original Image")
    axes[0].set_xlabel(question, fontsize=10, wrap=True)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    axes[1].imshow(masked_img)
    axes[1].set_title(f"After Pruning ({kept_mask.sum()}/{initial_visual_count} tokens kept)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig("masked_patches_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved to masked_patches_visualization.png")
    plt.show()

