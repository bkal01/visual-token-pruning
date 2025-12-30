import torch

from dataclasses import dataclass
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

@dataclass
class VLMInferenceResult():
    input_ids: torch.Tensor
    token_types: torch.Tensor
    captured_attentions: torch.Tensor
    generated_ids: torch.Tensor = None


def run_inference(
    model,
    processor,
    image,
    question,
    max_new_tokens=128,
):
    """
    Runs inference for a single sample. Does full generation.
    Returns:
    - input_ids: token ids of the input image & text
    - token_types: 0 for text tokens, 1 for visual tokens
    - generated_ids: token ids of the generated output
    - captured_attentions: attention scores for each layer
    """
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

    # reset captured attentions
    model.captured_attentions = []

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    # mark which tokens are visual tokens
    input_ids = inputs["input_ids"][0]
    token_types = torch.zeros_like(input_ids, device=device)
    token_types[input_ids == model.config.image_token_id] = 1

    return VLMInferenceResult(
        input_ids=input_ids,
        token_types=token_types,
        captured_attentions=torch.stack(model.captured_attentions).to(device),
        generated_ids=generated_ids,
    )


def run_prefill(
    model,
    processor,
    image,
    question,
):
    """
    Runs prefill for a single sample.
    Returns:
    - input_ids: token ids of the input image & text
    - token_types: 0 for text tokens, 1 for visual tokens
    - captured_attentions: attention scores for each layer
    """
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

    model.captured_attentions = []
    with torch.inference_mode():
        model.forward(**inputs)


    # mark which tokens are visual tokens
    input_ids = inputs["input_ids"][0]
    token_types = torch.zeros_like(input_ids, device=device)
    token_types[input_ids == model.config.image_token_id] = 1
    return VLMInferenceResult(
        input_ids=input_ids,
        token_types=token_types,
        captured_attentions=torch.stack(model.captured_attentions).to(device),
    )
