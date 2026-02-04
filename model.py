from dataclasses import dataclass

import torch
from transformers import AutoProcessor

from models.qwen3_vl.modeling_qwen3_vl import (
    InferenceContext,
    PrunedQwen3VL,
    VisionInferenceContext,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_name,
    pruner=None,
    rope_config=None,
):
    model = PrunedQwen3VL.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    model.set_pruner(pruner)
    model.set_rope_config(rope_config)

    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


@dataclass
class TimingInfo:
    ttft_ms: float = None
    decode_ms: float = None
    num_tokens: int = 0

    @property
    def total_ms(self):
        if self.ttft_ms is None or self.decode_ms is None:
            return None
        return self.ttft_ms + self.decode_ms

    @property
    def decode_latency_ms(self):
        if self.decode_ms is None or self.num_tokens == 0:
            return None
        return self.decode_ms / self.num_tokens


@dataclass
class VLMInferenceResult:
    input_ids: torch.Tensor
    image_grid_thw: torch.Tensor
    generated_ids: torch.Tensor = None
    inference_context: InferenceContext = None
    vision_inference_context: VisionInferenceContext = None
    timing: TimingInfo = None

    def get_surviving_indices(self):
        indices = []
        if (
            self.vision_inference_context
            and self.vision_inference_context.surviving_visual_indices
        ):
            indices.extend(self.vision_inference_context.surviving_visual_indices)
        if self.inference_context and self.inference_context.surviving_visual_indices:
            indices.extend(self.inference_context.surviving_visual_indices)
        return indices

    def get_pruning_ratio(self):
        indices = self.get_surviving_indices()
        if not indices:
            return 0.0
        return 1.0 - (len(indices[-1]) / len(indices[0]))

    def decode_output(self, processor):
        if self.generated_ids is None:
            return ""
        input_length = len(self.input_ids)
        output_ids = self.generated_ids[0, input_length:]
        return processor.decode(output_ids, skip_special_tokens=True)


class _TimingStreamer:
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.first_token_recorded = False
        self.token_count = 0

    def put(self, token_ids):
        if not self.first_token_recorded:
            self.first_token_event.record()
            self.first_token_recorded = True
        self.token_count += token_ids.shape[-1]

    def end(self):
        pass


def run_inference(
    model,
    processor,
    image,
    question,
    max_new_tokens=1024,
    timed=False,
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
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
    input_ids = inputs["input_ids"][0]

    image_grid_thw = inputs["image_grid_thw"][0]
    model.set_image_grid_thw(image_grid_thw)

    timing = None
    if timed and torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        streamer = _TimingStreamer()

        start_event.record()
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, streamer=streamer
        )
        end_event.record()
        torch.cuda.synchronize()

        ttft_ms = start_event.elapsed_time(streamer.first_token_event)
        total_ms = start_event.elapsed_time(end_event)
        num_tokens = generated_ids.shape[1] - len(input_ids)

        timing = TimingInfo(
            ttft_ms=ttft_ms,
            decode_ms=total_ms - ttft_ms,
            num_tokens=num_tokens,
        )
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return VLMInferenceResult(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        generated_ids=generated_ids,
        inference_context=model.model.language_model.inference_context,
        vision_inference_context=model.model.visual.inference_context,
        timing=timing,
    )


def reset_inference_context(model):
    model.model.language_model.inference_context.attentions.clear()
    model.model.language_model.inference_context.surviving_visual_indices.clear()
    model.model.visual.inference_context.attentions.clear()
    model.model.visual.inference_context.surviving_visual_indices.clear()


def run_prefill(model, processor, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
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
    input_ids = inputs["input_ids"][0]

    with torch.inference_mode():
        model.forward(**inputs)

    return VLMInferenceResult(
        input_ids=input_ids,
        image_grid_thw=inputs["image_grid_thw"][0],
        inference_context=model.model.language_model.inference_context,
        vision_inference_context=model.model.visual.inference_context,
    )
