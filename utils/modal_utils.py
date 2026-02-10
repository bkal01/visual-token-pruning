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


def download_model_and_data(model_name, dataset_name):
    """
    function that is run as a build step to download model/dataset.
    """
    from datasets import load_dataset
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"Downloading model: {model_name}")
    Qwen3VLForConditionalGeneration.from_pretrained(model_name)
    AutoProcessor.from_pretrained(model_name)

    print(f"Downloading dataset: {dataset_name}")
    for subset in DATBENCH_SUBSETS:
        print(f"Downloading subset: {subset}")
        load_dataset(dataset_name, subset, split="test")


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
            download_model_and_data,
            kwargs={"model_name": model_name, "dataset_name": dataset_name},
            secrets=[modal.Secret.from_name("huggingface")],
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
