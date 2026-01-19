import modal

from utils.modal_utils import get_modal_image

app = modal.App(name="template")
image = get_modal_image()

@app.function(image=image, gpu="A100", timeout=10*60)
def run():
    from datasets import load_dataset

    from model import load_model, run_inference
    from pruners.visiondrop import VisionDropPruner

    model, processor = load_model(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        pruner=VisionDropPruner(
            vision_target_layers=[23,],
            llm_target_layers=[8,16,24,27,],
            filtering_ratio=0.5,
        ),
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

    inference_context = model.get_inference_context()
    vision_inference_context = model.get_vision_inference_context()

    print(f"number of vision attention scores: {len(vision_inference_context.attentions)}")
    print(f"vision_inference_context.attentions[0].shape: {vision_inference_context.attentions[0].shape}")




@app.local_entrypoint()
def main():
    run.remote()