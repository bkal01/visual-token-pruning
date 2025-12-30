import modal

def get_modal_image():
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
                "datasets",
            ],
        )
        .add_local_dir(
            local_path="models",
            remote_path="/root/models",
        )
        .add_local_dir(
            local_path="utils",
            remote_path="/root/utils",
        )
        .add_local_file(
            local_path="visualization.py",
            remote_path="/root/visualization.py",
        )
        .add_local_file(
            local_path="model.py",
            remote_path="/root/model.py",
        )
    )
    return image