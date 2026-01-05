import modal
import numpy as np
import matplotlib.pyplot as plt

from utils.modal_utils import get_modal_image

app = modal.App(name="visualize-masked-patches")
image = get_modal_image()

LAYERS_TO_SHOW = [5, 10, 15, 20, 25]


@app.function(image=image, gpu="A100", timeout=10*60)
def run():
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

    dataset = iter(load_dataset("merve/vqav2-small", split="validation", streaming=True))
    sample = next(dataset)

    result = run_inference(
        model,
        processor,
        sample["image"],
        sample["question"],
    )

    all_surviving = result.inference_context.surviving_visual_indices
    initial_visual_count = len(all_surviving[0])
    
    layer_snapshots = {}
    for layer_idx in LAYERS_TO_SHOW:
        if layer_idx < len(all_surviving):
            layer_snapshots[layer_idx] = all_surviving[layer_idx].numpy()
    
    final_layer_idx = len(all_surviving) - 1
    layer_snapshots[final_layer_idx] = all_surviving[-1].numpy()

    return {
        "image": sample["image"],
        "question": sample["question"],
        "initial_visual_count": initial_visual_count,
        "layer_snapshots": layer_snapshots,
        "final_layer_idx": final_layer_idx,
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


def compute_grid_dims(initial_visual_count):
    grid_size = int(np.sqrt(initial_visual_count))
    if grid_size * grid_size != initial_visual_count:
        for gh in range(1, initial_visual_count + 1):
            if initial_visual_count % gh == 0:
                gw = initial_visual_count // gh
                if abs(gh - gw) < abs(grid_size - initial_visual_count // grid_size):
                    grid_size = gh
        return grid_size, initial_visual_count // grid_size
    return grid_size, grid_size


@app.local_entrypoint()
def main():
    data = run.remote()
    
    pil_image = data["image"]
    question = data["question"]
    initial_visual_count = data["initial_visual_count"]
    layer_snapshots = data["layer_snapshots"]
    final_layer_idx = data["final_layer_idx"]
    
    grid_h, grid_w = compute_grid_dims(initial_visual_count)
    print(f"Visual tokens: {initial_visual_count}, Grid: {grid_h}x{grid_w}")
    
    sorted_layers = sorted(layer_snapshots.keys())
    n_panels = 1 + len(sorted_layers)  # original + each layer snapshot
    
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    
    axes[0].imshow(pil_image)
    axes[0].set_title("Original")
    axes[0].set_xlabel(question, fontsize=8, wrap=True)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    for i, layer_idx in enumerate(sorted_layers):
        surviving = layer_snapshots[layer_idx]
        kept_mask = np.zeros(initial_visual_count, dtype=bool)
        kept_mask[surviving] = True
        
        masked_img = create_masked_image(pil_image, kept_mask, grid_h, grid_w)
        
        ax = axes[i + 1]
        ax.imshow(masked_img)
        
        if layer_idx == final_layer_idx:
            title = f"Final (L{layer_idx})"
        else:
            title = f"Layer {layer_idx}"
        ax.set_title(f"{title}\n{kept_mask.sum()}/{initial_visual_count} kept")
        ax.set_xticks([])
        ax.set_yticks([])
        
        print(f"Layer {layer_idx}: {kept_mask.sum()}/{initial_visual_count} tokens kept")
    
    plt.tight_layout()
    plt.savefig("masked_patches_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved to masked_patches_visualization.png")
    plt.show()

