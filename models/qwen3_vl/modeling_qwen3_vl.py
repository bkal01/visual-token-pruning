import torch

from dataclasses import dataclass
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLTextModel,Qwen3VLTextRotaryEmbedding
from transformers.modeling_rope_utils import dynamic_rope_update
from typing import List

@dataclass
class InferenceContext:
    """
    Captures layer-by-layer context during inference.
    attentions: the list of attention scores for each layer. entry i is a tensor of shape (T_i, T_i)
    token_types: the list of token types (1 for visual, 0 for text) for each layer. entry i is a tensor of shape (T_i,)
    kept_indices: the list of indices of the tokens that were kept at each layer. entry i is a tensor of shape (T_{i+1},)
    """
    attentions: List[torch.Tensor]
    token_types: List[torch.Tensor]
    kept_indices: List[torch.Tensor]


class PrunedQwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PrunedQwen3VLModel(self.config)


class PrunedQwen3VLModel(Qwen3VLModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language_model = PrunedQwen3VLTextModel(self.config)
        

class PrunedQwen3VLTextModel(Qwen3VLTextModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_context = InferenceContext(
            attentions=[],
            token_types=[],
            kept_indices=[],
        )
        self.register_hooks()

        if kwargs.get("use_standard_rope", False):
            print("Using standard RoPE for Qwen3-VL")
            self.rotary_emb = Qwen3VLTextStandardRotaryEmbedding(self.config)
        else:
            print("Using Interleaved MRoPE for Qwen3-VL")

    def register_hooks(self):
        def text_decoder_layer_forward_hook(module, input, output):
            if self.pruner is not None:
                layer_idx = module.self_attn.layer_idx
                output, token_types, kept_indices = self.pruner.prune(
                    layer_idx=layer_idx,
                    hidden_states=output,
                    attention_scores=self.inference_context.attentions[layer_idx],
                    token_types=self.inference_context.token_types[layer_idx],
                )
                self.inference_context.token_types.append(token_types.cpu())
                self.inference_context.kept_indices.append(kept_indices.cpu())
            return output

        def attn_forward_hook(module, input, output):
            attn_output, attn_weights = output
            if attn_weights.shape[2] > 1:
                self.inference_context.attentions.append(attn_weights.mean(dim=1).cpu())
            return output

        for idx, layer in enumerate(self.layers):
            layer.register_forward_hook(text_decoder_layer_forward_hook)
            layer.self_attn.register_forward_hook(attn_forward_hook)


    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return output

class Qwen3VLTextStandardRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = freqs[0]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)