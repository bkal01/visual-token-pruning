"""
This file is a modified version of the `modeling_qwen3_vl.py` file from the Hugging Face transfomers library.
See https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py for the original code.
Changes to forward passes are marked with [PRUNING MODIFICATION].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Optional, Union

from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import create_causal_mask
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import TransformersKwargs

from models.rope_config import RoPEConfig

@dataclass
class InferenceContext:
    """
    Captures context inside the LLM during inference that's useful for pruning.
    Allows us to pass information between the vision model and the text model.
    attentions: the list of attention scores for each layer. entry i is a tensor of shape (T_i, T_i)
    surviving_visual_indices: the list of indices of the visual tokens that were kept at each layer. entry i is a tensor of shape (V_{i+1},)
    image_grid_thw: the T,H,W dimensions of the input image.
    spatial_merge_size: how many patches get combined into a token via the VLM adapter.
    """
    attentions: List[torch.Tensor]
    surviving_visual_indices: List[torch.Tensor]
    image_grid_thw: torch.Tensor
    spatial_merge_size: int

@dataclass
class VisionInferenceContext:
    """
    Captures context inside the vision encoder during inference that's useful for pruning.
    """
    attentions: List[torch.Tensor]
    surviving_visual_indices: List[torch.Tensor]
    image_grid_thw: torch.Tensor
    spatial_merge_size: int

class PrunedQwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        old_model = self.model
        self.model = PrunedQwen3VLModel(config)
        self.model.load_state_dict(old_model.state_dict())

        old_vision = self.model.visual
        self.model.visual = PrunedQwen3VLVisionModel(config.vision_config)
        self.model.visual.load_state_dict(old_vision.state_dict())

        old_lm = self.model.language_model
        self.model.language_model = PrunedQwen3VLTextModel(config.text_config)
        self.model.language_model.load_state_dict(old_lm.state_dict())

        self.model.visual.inference_context.spatial_merge_size = config.vision_config.spatial_merge_size
        self.model.language_model.inference_context.spatial_merge_size = config.vision_config.spatial_merge_size

    def set_pruner(self, pruner):
        self.model.visual.pruner = pruner
        self.model.language_model.pruner = pruner

    def set_rope_config(self, rope_config):
        self.model.language_model.rope_config = rope_config

    def set_image_grid_thw(self, image_grid_thw):
        self.model.visual.inference_context.image_grid_thw = image_grid_thw
        self.model.language_model.inference_context.image_grid_thw = image_grid_thw

    def get_inference_context(self):
        return self.model.language_model.inference_context

    def get_vision_inference_context(self):
        return self.model.visual.inference_context

class PrunedQwen3VLModel(Qwen3VLModel):
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # [PRUNING MODIFICATION] we need to modify this function to handle image_embeds changing shape from pruning.
        # since we're assuming B=1, we can just wrap image_embeds in a tuple to simulate what torch.split would do.
        # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        # image_embeds = torch.split(image_embeds, split_sizes)
        return (image_embeds,), deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        # [PRUNING MODIFICATION] we ignore this check because image_features might be smaller.
        # if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
        #     raise ValueError(
        #         f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        #     )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if position_ids is None:
            past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
            if self.rope_deltas is None or past_key_values_length == 0:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (past_key_values_length + self.rope_deltas).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )

            surviving_visual_indices = self.visual.inference_context.surviving_visual_indices[-1].to(inputs_embeds.device)
            surviving_token_indices = torch.unique(surviving_visual_indices // self.visual.inference_context.spatial_merge_size**2)
            num_remaining_visual_tokens = len(surviving_token_indices)
            visual_indices = torch.where(image_mask[0].all(dim=1))[0]
            start, end = visual_indices[0], visual_indices[-1] + 1

            image_mask = torch.cat([
                image_mask[:, :start],
                image_mask[:, start:start + num_remaining_visual_tokens],
                image_mask[:, end:],
            ], dim=1)
            input_ids = torch.cat([
                input_ids[:, :start],
                input_ids[:, start:start + num_remaining_visual_tokens],
                input_ids[:, end:],
            ], dim=1)
            visual_inputs_embeds = inputs_embeds[:, start:end, :]
            inputs_embeds = torch.cat([
                inputs_embeds[:, :start],
                visual_inputs_embeds[:, surviving_token_indices, :],
                inputs_embeds[:, end:],
            ], dim=1)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask[:, :start],
                    attention_mask[:, start:start + num_remaining_visual_tokens],
                    attention_mask[:, end:],
                ], dim=1)
            position_ids = torch.cat([
                position_ids[:, :, :start],
                position_ids[:, :, start:start + num_remaining_visual_tokens],
                position_ids[:, :, end:],
            ], dim=2)

            if cache_position is not None:
                cache_position = torch.cat([
                    cache_position[:start],
                    cache_position[start:start + num_remaining_visual_tokens],
                    cache_position[end:]
                ])

            if deepstack_image_embeds is not None:
                deepstack_image_embeds = [
                    embed[surviving_token_indices, :] for embed in deepstack_image_embeds
                ]

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )



class PrunedQwen3VLVisionModel(Qwen3VLVisionModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        old_blocks = self.blocks
        self.blocks = nn.ModuleList([PrunedQwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.blocks.load_state_dict(old_blocks.state_dict())

        self.pruner = None
        self.inference_context = VisionInferenceContext(
            attentions=[],
            surviving_visual_indices=[],
            image_grid_thw=None,
            spatial_merge_size=0,
        )

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        surviving_visual_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        self.inference_context.surviving_visual_indices.append(surviving_visual_indices.cpu())

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states, layer_attn_weights = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            avg_layer_attn_weights = layer_attn_weights.mean(dim=1).squeeze(0) # assume B=1 for now.
            self.inference_context.attentions.append(avg_layer_attn_weights.cpu())
            if self.pruner is not None:
                hidden_states, keep_mask = self.pruner.prune_vision_forward(
                    layer_idx=layer_num,
                    hidden_states=hidden_states,
                    attention_scores=avg_layer_attn_weights,
                    spatial_merge_size=self.config.spatial_merge_size,
                )
                if not keep_mask.all():
                    surviving_visual_indices = surviving_visual_indices[keep_mask]
                    self.inference_context.surviving_visual_indices.append(surviving_visual_indices.cpu())

            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists

class PrunedQwen3VLVisionBlock(Qwen3VLVisionBlock):
    def __init__(self, config):
        super().__init__(config)
        self.attn = PrunedQwen3VLVisionAttention(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_output, attn_weights = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, attn_weights

class PrunedQwen3VLVisionAttention(Qwen3VLVisionAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Non-Flash Attention Path
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_interface_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )
            for q, k, v in zip(*splits)
        ]
        attn_outputs, attn_weights = [attn_interface_output[0] for attn_interface_output in attn_interface_outputs], [attn_interface_output[1] for attn_interface_output in attn_interface_outputs]
        # assuming B=1, these cats should do nothing.
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_weights = torch.cat(attn_weights, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output, attn_weights

class PrunedQwen3VLTextModel(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.pruner = None
        self.inference_context = InferenceContext(
            attentions=[],
            surviving_visual_indices=[],
            image_grid_thw=None,
            spatial_merge_size=0,
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
        # [PRUNING MODIFICATION] apply RoPE config.
        if self.rope_config == RoPEConfig.NONE:
            batch_size, seq_len, _ = hidden_states.shape
            head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
            position_embeddings = (
                torch.zeros(batch_size, seq_len, head_dim, device=hidden_states.device, dtype=hidden_states.dtype),
                torch.zeros(batch_size, seq_len, head_dim, device=hidden_states.device, dtype=hidden_states.dtype),
            )
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        is_prefill = hidden_states.shape[1] > 1
        if is_prefill and visual_pos_masks is not None:
            surviving_visual_indices = torch.arange(visual_pos_masks[0].sum().item(), device=visual_pos_masks.device)
            self.inference_context.surviving_visual_indices.append(surviving_visual_indices.cpu())

        # [PRUNING MODIFICATION] pruning via prune_embeddings().
        if is_prefill and self.pruner is not None:
            hidden_states, keep_mask = self.pruner.prune_embeddings(
                embeddings=hidden_states,
                token_types=visual_pos_masks[0],
            )
            visual_keep_mask = keep_mask[visual_pos_masks[0]]
            if not visual_keep_mask.all():
                surviving_visual_indices = surviving_visual_indices[visual_keep_mask]
                self.inference_context.surviving_visual_indices.append(surviving_visual_indices.cpu())

            visual_pos_masks = visual_pos_masks[:, keep_mask]
            cos, sin = position_embeddings
            position_embeddings = (cos[:, keep_mask, :], sin[:, keep_mask, :])
            attention_mask = attention_mask[:, :, keep_mask, :][:, :, :, keep_mask]
            text_position_ids = text_position_ids[:, keep_mask]
            cache_position = cache_position[keep_mask]

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
                avg_layer_attn_weights = layer_attn_weights.mean(dim=1).squeeze(0) # assume B=1 for now.
                self.inference_context.attentions.append(avg_layer_attn_weights.cpu())

                if self.pruner is not None:
                    hidden_states, keep_mask = self.pruner.prune_decoder_forward(
                        layer_idx=layer_idx,
                        hidden_states=hidden_states,
                        token_types=visual_pos_masks[0],
                        attention_scores=avg_layer_attn_weights,
                        image_grid_thw=self.inference_context.image_grid_thw,
                        spatial_merge_size=self.inference_context.spatial_merge_size,
                    )
                    visual_keep_mask = keep_mask[visual_pos_masks[0]]
                    if not visual_keep_mask.all():
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