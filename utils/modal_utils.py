import modal

DATBENCH_SUBSETS = [
    "chart",
    "counting",
    "document",
    "general",
    "grounding",
    "math",
    "scene",
    "spatial",
    "table",
]


def download_data(dataset_name):
    import os

    from datasets import load_dataset

    print(f"Downloading dataset: {dataset_name}")
    for subset in DATBENCH_SUBSETS:
        print(f"Downloading subset: {subset}")
        ds = load_dataset(dataset_name, subset, split="test")
        filtered = ds.filter(lambda x: x["eval_mode"] == "direct")
        save_path = f"/root/datbench_filtered/{subset}"
        os.makedirs(save_path, exist_ok=True)
        filtered.save_to_disk(save_path)
        print(f"  Saved {len(filtered)} filtered samples to {save_path}")


def download_model(model_name):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"Downloading model: {model_name}")
    Qwen3VLForConditionalGeneration.from_pretrained(model_name)
    AutoProcessor.from_pretrained(model_name)


def get_modal_image(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    dataset_name="DatologyAI/DatBench",
):
    image = (
        modal.Image.debian_slim(
            python_version="3.10",
        )
        .apt_install(
            "git",
        )
        .env(
            {
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            }
        )
        .uv_pip_install(
            [
                "transformers>=4.57.0",
                "torch>=2.6.0",
                "torchvision>=0.15.0",
                "pillow>=10.0.0",
                "matplotlib",
                "datasets>=2.14.0",
                "pyarrow>=14.0.0,<19.0.0",
                "fsspec>=2023.10.0,<2024.12.0",
                "pandas",
                "accelerate",
                "datbench @ git+https://github.com/bkal01/DatBench.git@fix/boxed-parsing",
            ],
        )
        .run_function(
            download_data,
            kwargs={"dataset_name": dataset_name},
            secrets=[modal.Secret.from_name("huggingface")],
            timeout=45 * 60,
        )
        .run_function(
            download_model,
            kwargs={"model_name": model_name},
            secrets=[modal.Secret.from_name("huggingface")],
            timeout=45 * 60,
        )
        .add_local_dir(
            local_path="models",
            remote_path="/root/models",
        )
        .add_local_dir(
            local_path="utils",
            remote_path="/root/utils",
        )
        .add_local_dir(
            local_path="pruners",
            remote_path="/root/pruners",
        )
        .add_local_dir(
            local_path="eval",
            remote_path="/root/eval",
        )
        .add_local_file(
            local_path="model.py",
            remote_path="/root/model.py",
        )
    )
    return image
