"""
This file is a modified version of the `modeling_qwen3_vl.py` file from the Hugging Face transfomers library.
See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py for the original code.
Changes to forward passes are marked with [PRUNING MODIFICATION].
"""
import torch

from dataclasses import dataclass
from typing import List, Optional, Union

from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer,Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import TransformersKwargs

@dataclass
class InferenceContext:
    """
    Captures layer-by-layer context during inference.
    attentions: the list of attention scores for each layer. entry i is a tensor of shape (T_i, T_i)
    surviving_visual_indices: the list of indices of the visual tokens that were kept at each layer. entry i is a tensor of shape (V_{i+1},)
    """
    attentions: List[torch.Tensor]
    surviving_visual_indices: List[torch.Tensor]


class PrunedQwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        old_lm = self.model.language_model
        self.model.language_model = PrunedQwen3VLTextModel(config.text_config)
        self.model.language_model.load_state_dict(old_lm.state_dict())

    def set_pruner(self, pruner):
        self.model.language_model.pruner = pruner


class PrunedQwen3VLTextModel(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.pruner = None
        self.inference_context = InferenceContext(
            attentions=[],
            surviving_visual_indices=[],
        )
        for i, layer in enumerate(self.layers):
            new_layer = PrunedQwen3VLTextDecoderLayer(config, i)
            new_layer.load_state_dict(layer.state_dict())
            self.layers[i] = new_layer

    # copied from https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L826
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        is_prefill = hidden_states.shape[1] > 1
        if is_prefill and visual_pos_masks is not None:
            surviving_visual_indices = torch.arange(visual_pos_masks[0].sum().item(), device=visual_pos_masks.device)
            self.inference_context.surviving_visual_indices.append(surviving_visual_indices.cpu())

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs, layer_attn_weights = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs
            
            # [PRUNING MODIFICATION] pruning.
            if is_prefill and visual_pos_masks is not None:
                avg_layer_attn_weights = layer_attn_weights.mean(dim=1).squeeze(0)
                self.inference_context.attentions.append(avg_layer_attn_weights.cpu())

                if self.pruner is not None:
                    hidden_states, keep_mask = self.pruner.prune(
                        layer_idx=layer_idx,
                        hidden_states=hidden_states,
                        attention_scores=avg_layer_attn_weights,
                        token_types=visual_pos_masks[0],
                    )
                    visual_keep_mask = keep_mask[visual_pos_masks[0]]
                    surviving_visual_indices = surviving_visual_indices[visual_keep_mask]
                    self.inference_context.surviving_visual_indices.append(surviving_visual_indices.cpu())

                    visual_pos_masks = visual_pos_masks[:, keep_mask]
                    cos, sin = position_embeddings
                    position_embeddings = (cos[:, keep_mask, :], sin[:, keep_mask, :])
                    attention_mask = attention_mask[:, :, keep_mask, :][:, :, :, keep_mask]
                    text_position_ids = text_position_ids[:, keep_mask]
                    cache_position = cache_position[keep_mask]

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                if is_prefill and self.pruner is not None:
                    deepstack_visual_embeds[layer_idx] = deepstack_visual_embeds[layer_idx][surviving_visual_indices, :]
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

class PrunedQwen3VLTextDecoderLayer(Qwen3VLTextDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        # [PRUNING MODIFICATION] return the attention weights
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        # [PRUNING MODIFICATION] return the attention weights
        return hidden_states, attn_weights


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