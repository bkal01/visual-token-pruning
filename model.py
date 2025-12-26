import torch
from transformers import AutoProcessor

from models.qwen3_vl.modeling_qwen3_vl import PrunedQwen3VL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    model = PrunedQwen3VL.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="eager"
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def run_inference(
    model,
    processor,
    image,
    question,
    max_new_tokens=128,
):
    messages = [
        {
            "role":"user",
            "content":[
                {
                    "type":"image",
                    "image": image,
                },
                {
                    "type":"text",
                    "text":question,
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
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    # mark which tokens are visual tokens
    input_ids = inputs["input_ids"][0]
    token_types = torch.zeros_like(input_ids)
    token_types[input_ids == model.config.image_token_id] = 1

    return inputs["input_ids"], token_types, generated_ids, model.captured_attentions