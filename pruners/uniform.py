import torch

from pruners.base_pruner import Pruner

class UniformPruner(Pruner):
    def __init__(
        self,
        target_layers,
        stride,
    ):
        super().__init__(target_layers)
        self.stride = stride

    def prune(
        self,
        layer_idx,
        hidden_states,
        attention_scores,
        token_types,
        image_grid_thw,
        spatial_merge_size,
    ):
        """
        Prunes visual tokens uniformly using a stride.
        NOTE: this only works when the image is being pruned for the first time, it will fail
        if the image has been pruned in a previous layer (since len(token_types) < H_tok * W_tok).
        This is fine for now since most pruning methods do uniform sampling just once at the first pruning instance.
        """
        T = attention_scores.shape[0]
        if layer_idx not in self.target_layers:
            return hidden_states, torch.ones(T, dtype=torch.bool, device=token_types.device)

        keep_mask = ~token_types.bool()
        visual_indices = token_types.nonzero(as_tuple=True)[0]
        H_tok, W_tok = image_grid_thw[1] // spatial_merge_size, image_grid_thw[2] // spatial_merge_size
        visual_indices_2d = visual_indices.view(H_tok, W_tok)
        visual_indices_2d = visual_indices_2d[::self.stride, ::self.stride]
        keep_mask[visual_indices_2d.flatten()] = True

        return hidden_states[:, keep_mask, :], keep_mask