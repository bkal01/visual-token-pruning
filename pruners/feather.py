import torch

from pruners.base_pruner import Pruner

class FeatherPruner(Pruner):
    def __init__(
        self,
        target_layers,
        random_target_layers,
        filtering_ratio,
    ):
        super().__init__(target_layers)
        
        self.random_target_layers = random_target_layers
        self.filtering_ratio = filtering_ratio


    def prune(
        self,
        layer_idx,
        hidden_states,
        attention_scores,
        token_types,
    ):
        """
        Prunes tokens using FEATHER.
        For layers in `self.target_layers`, prune tokens according to the attention received from the last text token.
        For layers in `self.random_target_layers`, prune tokens uniformly at random.
        If `layer_idx` is in both, then we keep the union of tokens selected by both methods.
        """
        T = attention_scores.shape[0]
        V = int(token_types.sum())

        if layer_idx not in self.target_layers and layer_idx not in self.random_target_layers:
            return hidden_states, torch.ones(T, dtype=torch.bool, device=token_types.device)
        return hidden_states, torch.ones(T, dtype=torch.bool, device=token_types.device)
