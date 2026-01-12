import modal
import numpy as np
import matplotlib.pyplot as plt
import os

from uuid import uuid4

from utils.modal_utils import get_modal_image

app = modal.App(name="visualize-masked-patches")
image = get_modal_image()


@app.function(image=image, gpu="A100", timeout=10*60)
def run():
    from datasets import load_dataset

    from model import load_model, run_inference
    from pruners.fastv_pruner import FastVPruner
    from pruners.feather import FeatherPruner
    from pruners.uniform_pruner import UniformPruner

    # pruner = FastVPruner(
    #     target_layers=[1], # FastV paper prunes after layer 2's forward pass.
    #     filtering_ratio=0.5,
    # )
    # pruner = FeatherPruner(
    #     target_layers=[8,16,],
    #     uniform_target_layers=[8,],
    #     filtering_ratio=0.75,
    #     stride=3,
    # )
    pruner = UniformPruner(
        target_layers=[3],
        stride=3,
    )
    model, processor = load_model(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        pruner=pruner,
        rope_config=None,
    )

    dataset = iter(load_dataset("DatologyAI/DatBench", "math", split="test", streaming=True))
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

    return sample, inference_context.surviving_visual_indices, result.image_grid_thw.cpu(), spatial_merge_size, patch_size, pruner.target_layers




@app.local_entrypoint()
def main():
    id = str(uuid4())
    print(f"Run UUID: {id}")
    os.makedirs(f"assets/visualize_masked_patches/{id}/", exist_ok=True)
    sample, surviving_visual_indices, image_grid_thw, spatial_merge_size, patch_size, target_layers = run.remote()

    print(f"QUESTION: {sample['question']}")

    H_tok, W_tok = image_grid_thw[1] // spatial_merge_size, image_grid_thw[2] // spatial_merge_size

    image_np = np.array(sample["image"])
    plt.imshow(image_np.astype(np.uint8))
    plt.axis("off")
    plt.savefig(f"assets/visualize_masked_patches/{id}/original.png")
    plt.close()


    token_size = spatial_merge_size * patch_size

    for layer_idx in target_layers:
        copy_image_np = image_np.copy() * 0.1
        for visual_token_index in surviving_visual_indices[layer_idx + 1]:
            row, col = visual_token_index // W_tok, visual_token_index % W_tok
            pixel_start_row = row * token_size
            pixel_start_col = col * token_size

            copy_image_np[pixel_start_row:pixel_start_row + token_size,
                          pixel_start_col:pixel_start_col + token_size] = \
                image_np[pixel_start_row:pixel_start_row + token_size,
                          pixel_start_col:pixel_start_col + token_size]

        fig, ax = plt.subplots()
        ax.imshow(copy_image_np.astype(np.uint8))
        ax.set_xticks(np.arange(0, image_np.shape[1], token_size))
        ax.set_yticks(np.arange(0, image_np.shape[0], token_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color="gray", alpha=0.3, linewidth=0.5)
        ax.tick_params(length=0)
        plt.savefig(f"assets/visualize_masked_patches/{id}/layer_{layer_idx + 1}.png", bbox_inches='tight', pad_inches=0)
        plt.close()