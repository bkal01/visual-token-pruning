import modal
import numpy as np
import matplotlib.pyplot as plt
import os

from uuid import uuid4

from utils.modal_utils import get_modal_image

app = modal.App(name="vision-attention")
image = get_modal_image()

@app.function(image=image, gpu="A100", timeout=10*60)
def run():
    from datasets import load_dataset

    from model import load_model, run_inference
    from pruners.visiondrop import VisionDropPruner


    pruner = VisionDropPruner(
        vision_target_layers=[],
        llm_target_layers=[8,16,24,27],
        filtering_ratio=0.2,
    )
    model, processor = load_model(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
    )
    model.model.visual.pruner = pruner
    model.model.language_model.pruner = pruner
    dataset = iter(load_dataset("DatologyAI/DatBench", "math", split="test", streaming=True))
    sample = next(dataset)

    result = run_inference(
        model,
        processor,
        sample["image"],
        sample["question"],
        max_new_tokens=128,
    )

    generated_ids_trimmed = result.generated_ids[:, result.input_ids.shape[0]:]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    print(f"Question: {sample['question']}")
    print(f"Model output: {output_text}")

    vision_inference_context = result.vision_inference_context
    inference_context = result.inference_context

    spatial_merge_size = model.model.visual.config.spatial_merge_size
    patch_size = model.model.visual.config.patch_size

    return (
        sample,
        vision_inference_context.surviving_visual_indices,
        inference_context.surviving_visual_indices,
        result.image_grid_thw.cpu(),
        spatial_merge_size,
        patch_size,
    )




@app.local_entrypoint()
def main():
    id = str(uuid4())
    print(f"Run UUID: {id}")
    os.makedirs(f"assets/vision_attention/{id}/", exist_ok=True)
    (
        sample,
        vision_surviving_visual_indices,
        llm_surviving_visual_indices,
        image_grid_thw,
        spatial_merge_size,
        patch_size,
    ) = run.remote()

    print(f"QUESTION: {sample['question']}")

    H_tok, W_tok = image_grid_thw[1] // spatial_merge_size, image_grid_thw[2] // spatial_merge_size

    image_np = np.array(sample["image"])
    plt.imshow(image_np.astype(np.uint8))
    plt.axis("off")
    plt.savefig(f"assets/vision_attention/{id}/original.png")
    plt.close()

    token_size = spatial_merge_size * patch_size

    # Visualize vision encoder pruning steps
    for step_idx in range(1, len(vision_surviving_visual_indices)):
        copy_image_np = image_np.copy() * 0.1
        for visual_token_index in vision_surviving_visual_indices[step_idx]:
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
        plt.savefig(f"assets/vision_attention/{id}/vision_step_{step_idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    # Visualize LLM decoder pruning steps
    for step_idx in range(1, len(llm_surviving_visual_indices)):
        copy_image_np = image_np.copy() * 0.1
        for visual_token_index in llm_surviving_visual_indices[step_idx]:
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
        plt.savefig(f"assets/vision_attention/{id}/llm_step_{step_idx}.png", bbox_inches='tight', pad_inches=0)
        plt.close()