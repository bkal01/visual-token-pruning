import torch

from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from transformers.modeling_rope_utils import dynamic_rope_update

class PrunedQwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_attentions = []
        self.register_hooks()

        if kwargs.get("use_standard_rope", False):
            print("Using standard RoPE for Qwen3-VL")
            self.model.language_model.rotary_emb = Qwen3VLTextStandardRotaryEmbedding(self.model.language_model.config)
        else:
            print("Using Interleaved MRoPE for Qwen3-VL")

    def register_hooks(self):
        """
        hook into Qwen3VLTextAttention and capture the attention weights during prefill
        """
        def hook(module, input, output):
            attn_output, attn_weights = output
            if attn_weights.shape[2] > 1:
                self.captured_attentions.append(attn_weights.cpu())
            return output

        for idx, layer in enumerate(self.model.language_model.layers):
            layer.self_attn.register_forward_hook(hook)


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