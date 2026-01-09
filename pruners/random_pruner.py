import torch

from pruners.base_pruner import Pruner

class RandomPruner(Pruner):
    def __init__(
        self,
        target_layers,
        filtering_ratio,
    ):
        super().__init__(target_layers)
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
        T = attention_scores.shape[0]
        V = int(token_types.sum())
        if layer_idx not in self.target_layers:
            return hidden_states, torch.ones(T, dtype=torch.bool, device=token_types.device)

        amount_to_keep = int(V * (1 - self.filtering_ratio))
        visual_indices = token_types.nonzero(as_tuple=True)[0]
        keep_mask = ~token_types.bool()
        keep_mask[visual_indices[torch.randperm(V, device=token_types.device)[:amount_to_keep]]] = True
        return hidden_states[:, keep_mask, :], keep_mask