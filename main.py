import modal

app = modal.App(name="visual-token-pruning")

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
        ],
    )
)

@app.function(image=image, gpu="A100")
def run_vlm():
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU configuration.")
        return
    device = torch.device("cuda")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype=torch.float16,
        attn_implementation="sdpa"
    ).to(device)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    messages = [
        {
            "role":"user",
            "content":[
                {
                    "type":"image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
                },
                {
                    "type":"text",
                    "text":"Describe this image."
                }
            ]
        }

    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)


@app.local_entrypoint()
def main():
    run_vlm.remote()
