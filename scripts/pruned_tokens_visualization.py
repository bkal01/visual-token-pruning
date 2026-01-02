"""
Script to visualize what tokens get pruned by a specific pruning method.
"""
import matplotlib.pyplot as plt
import modal
import os

from uuid import uuid4

from utils.modal_utils import get_modal_image

app = modal.App(name="pruned-tokens-visualization")
image = get_modal_image()

@app.function(image=image, gpu="A100", timeout=600)
def run(
    model_name,
    dataset,
    split,
):
    import torch
    from datasets import load_dataset

    from model import load_model, run_prefill
    from pruners.fastv_pruner import FastVPruner

    device = torch.device("cuda")

    pruner = FastVPruner(
        layer_threshold=0,
        filtering_ratio=0.2,
    )
    model, processor = load_model(
        model_name,
        pruner=pruner,
    )

    dataset_iterator = iter(load_dataset(dataset, split=split, streaming=True))

    sample = next(dataset_iterator)
    result =run_prefill(
        model,
        processor,
        sample["image"],
        sample["question"],
    )

    print(len(result.inference_context.attentions))
    print(len(result.inference_context.token_types))
    print(len(result.inference_context.kept_indices))

    print(result.inference_context.attentions[0].shape)
    print(result.inference_context.token_types[0].shape)
    print(result.inference_context.kept_indices[0].shape)

    print(result.inference_context.attentions[-1].shape)
    print(result.inference_context.token_types[-1].shape)
    print(result.inference_context.kept_indices[-1].shape)


@app.local_entrypoint()
def main(
    model_name="Qwen/Qwen3-VL-2B-Instruct",
    dataset="merve/vqav2-small",
    split="validation",
):
    id = str(uuid4())
    print(f"Run UUID: {id}")

    run.remote(
        model_name=model_name,
        dataset=dataset,
        split="validation",
    )