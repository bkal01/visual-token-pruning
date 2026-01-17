import modal

from utils.modal_utils import get_modal_image

app = modal.App(name="template")
image = get_modal_image()

@app.function(image=image, gpu="A100", timeout=10*60)
def run():
    from datasets import load_dataset
    from pruners.flashvlm import FlashVLMPruner

    from model import load_model, run_inference

    model, processor = load_model(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        pruner=FlashVLMPruner(filtering_ratio=0.5),
        rope_config=None,
    )

    dataset = iter(load_dataset("DatologyAI/DatBench", "math", split="test", streaming=True))
    sample = next(dataset)

    run_inference(
        model,
        processor,
        sample["image"],
        sample["question"],
    )


@app.local_entrypoint()
def main():
    run.remote()