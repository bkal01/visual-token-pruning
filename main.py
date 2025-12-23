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
            "matplotlib"
        ],
    )
    .add_local_dir(
        local_path="models",
        remote_path="/root/models",
    )
    .add_local_file(
        local_path="visualization.py",
        remote_path="/root/visualization.py",
    )
)

@app.function(image=image, gpu="A100")
def run_vlm():
    import torch
    from transformers import AutoProcessor

    from models.qwen3_vl.modeling_qwen3_vl import PrunedQwen3VL
    from visualization import plot_attn_heatmap

    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your GPU configuration.")
        return
    device = torch.device("cuda")

    model = PrunedQwen3VL.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype=torch.float16,
        attn_implementation="eager"
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

    num_input_tokens = len(inputs["input_ids"][0])
    print(f"number of input tokens: {num_input_tokens}")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
    )
    prefill_attn_scores = model.captured_attentions[:28]
    layer_1_scores, layer_14_scores, layer_28_scores = prefill_attn_scores[0], prefill_attn_scores[13], prefill_attn_scores[27]
    print(f"layer & prefill attention score shape: (1, {layer_1_scores.shape}), (14, {layer_14_scores.shape}), (28, {layer_28_scores.shape})")
    
    # average across heads
    layer_1_scores, layer_14_scores, layer_28_scores = layer_1_scores.mean(dim=1).squeeze(dim=0), layer_14_scores.mean(dim=1).squeeze(dim=0), layer_28_scores.mean(dim=1).squeeze(dim=0)
    layer_1_img_bytes, layer_14_img_bytes, layer_28_img_bytes = plot_attn_heatmap(layer_1_scores), plot_attn_heatmap(layer_14_scores), plot_attn_heatmap(layer_28_scores)


    generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

    return layer_1_img_bytes, layer_14_img_bytes, layer_28_img_bytes


@app.local_entrypoint()
def main():
    layer_1_img_bytes, layer_14_img_bytes, layer_28_img_bytes = run_vlm.remote()
    with open("assets/layer_1_attention_heatmap.png", "wb") as f:
        f.write(layer_1_img_bytes)
    print("Saved assets/layer_1_attention_heatmap.png")
    with open("assets/layer_14_attention_heatmap.png", "wb") as f:
        f.write(layer_14_img_bytes)
    print("Saved assets/layer_14_attention_heatmap.png")
    with open("assets/layer_28_attention_heatmap.png", "wb") as f:
        f.write(layer_28_img_bytes)
    print("Saved assets/layer_28_attention_heatmap.png")
