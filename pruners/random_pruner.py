import torch

from pruners.base_pruner import Pruner

class RandomPruner(Pruner):
    def __init__(
        self,
        layer_threshold,
        filtering_ratio,
    ):
        super().__init__()
        self.layer_threshold = layer_threshold
        self.filtering_ratio = filtering_ratio

    def prune(
        self,
        layer_idx,
        hidden_states,
        attention_scores,
        token_types,
    ):
        """
        Prunes visual tokens uniformly at random.
        """
        if layer_idx < self.layer_threshold:
            return hidden_states, token_types
        V = int(token_types.sum())

        amount_to_keep = int(V * (1 - self.filtering_ratio))
        visual_indices = token_types.nonzero(as_tuple=True)[0]
        keep_mask = ~token_types.bool()
        keep_mask[visual_indices[torch.randperm(V, device=token_types.device)[:amount_to_keep]]] = True
        return hidden_states[:, keep_mask, :], token_types[keep_mask]