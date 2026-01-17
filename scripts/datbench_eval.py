import modal

from utils.modal_utils import get_modal_image

app = modal.App(name="datbench-eval")
image = get_modal_image()

@app.function(image=image, gpu="A100", timeout=10*60)
def run():
    pass


@app.local_entrypoint()
def main():
    run.remote()