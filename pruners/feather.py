import torch

from models.rope_config import RoPEConfig
from pruners.base_pruner import Pruner

class FeatherPruner(Pruner):
    def __init__(
        self,
        target_layers,
        uniform_target_layers,
        filtering_ratio,
        stride,
        rope_override=None,
    ):
        self.target_layers = target_layers
        self.uniform_target_layers = uniform_target_layers
        self.filtering_ratio = filtering_ratio
        self.stride = stride

        # FEATHER removes RoPE from the model to avoid effects of long-term decay on attention scores.
        if rope_override:
            self.rope_config = rope_override
        else:
            self.rope_config = RoPEConfig.NONE


    def prune_decoder_forward(
        self,
        layer_idx,
        hidden_states,
        token_types,
        **kwargs,
    ):
        """
        Prunes tokens using FEATHER.
        For layers in `self.target_layers`, prune tokens according to the attention received from the last text token.
        For layers in `self.uniform_target_layers`, prune tokens uniformly with stride `self.stride`.
        If `layer_idx` is in both, then we keep the union of tokens kept by both methods.
        """
        attention_scores = kwargs["attention_scores"]
        image_grid_thw = kwargs["image_grid_thw"]
        spatial_merge_size = kwargs["spatial_merge_size"]

        T = len(token_types)
        V = int(token_types.sum())

        if layer_idx not in self.target_layers and layer_idx not in self.uniform_target_layers:
            return hidden_states, torch.ones(T, dtype=torch.bool, device=token_types.device)

        keep_mask = ~token_types.bool()
        visual_indices = token_types.nonzero(as_tuple=True)[0]

        if layer_idx in self.target_layers:
            last_text_token_attention_scores = attention_scores[-1, :]
            visual_token_scores = last_text_token_attention_scores[token_types == 1]

            amount_to_keep = int(V * (1 - self.filtering_ratio))
            topk_relative = visual_token_scores.topk(amount_to_keep).indices.sort().values

            keep_mask[visual_indices[topk_relative]] = True

        if layer_idx in self.uniform_target_layers:
            H_tok, W_tok = image_grid_thw[1] // spatial_merge_size, image_grid_thw[2] // spatial_merge_size
            visual_indices_2d = visual_indices.view(H_tok, W_tok)
            visual_indices_2d = visual_indices_2d[::self.stride, ::self.stride]
            keep_mask[visual_indices_2d.flatten()] = True
        
        return hidden_states[:, keep_mask, :], keep_mask
