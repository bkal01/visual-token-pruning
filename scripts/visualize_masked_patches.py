import modal
import numpy as np
import matplotlib.pyplot as plt
import os

from uuid import uuid4

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

    inference_context = model.get_inference_context()

    spatial_merge_size = model.model.visual.config.spatial_merge_size
    patch_size = model.model.visual.config.patch_size

    return sample, inference_context.surviving_visual_indices, result.image_grid_thw.cpu(), spatial_merge_size, patch_size




@app.local_entrypoint()
def main():
    id = str(uuid4())
    print(f"Run UUID: {id}")
    os.makedirs(f"assets/visualize_masked_patches/{id}/", exist_ok=True)
    sample, surviving_visual_indices, image_grid_thw, spatial_merge_size, patch_size = run.remote()

    H_tok, W_tok = image_grid_thw[1] // spatial_merge_size, image_grid_thw[2] // spatial_merge_size

    image_np = np.array(sample["image"])

    for layer_idx in LAYERS_TO_SHOW:
        copy_image_np = image_np.copy() * 0.1
        for visual_token_index in surviving_visual_indices[layer_idx]:
            row, col = visual_token_index // W_tok, visual_token_index % W_tok
            pixel_start_row = row * spatial_merge_size * patch_size
            pixel_start_col = col * spatial_merge_size * patch_size

            copy_image_np[pixel_start_row:pixel_start_row + spatial_merge_size * patch_size,
                          pixel_start_col:pixel_start_col + spatial_merge_size * patch_size] = \
                image_np[pixel_start_row:pixel_start_row + spatial_merge_size * patch_size,
                          pixel_start_col:pixel_start_col + spatial_merge_size * patch_size]

        plt.imshow(copy_image_np.astype(np.uint8))
        plt.xlabel(sample["question"])
        plt.savefig(f"assets/visualize_masked_patches/{id}/layer_{layer_idx}.png")
        plt.close()