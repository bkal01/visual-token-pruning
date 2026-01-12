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
    ):
        """
        Prunes visual tokens uniformly using a stride.
        """
        T = attention_scores.shape[0]
        if layer_idx not in self.target_layers:
            return hidden_states, torch.ones(T, dtype=torch.bool, device=token_types.device)

        keep_mask = ~token_types.bool()
        visual_indices = token_types.nonzero(as_tuple=True)[0]
        keep_mask[visual_indices[::self.stride]] = True

        return hidden_states[:, keep_mask, :], keep_mask