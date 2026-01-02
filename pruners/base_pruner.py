import torch

from abc import ABC

class Pruner(ABC):
    """
    Base class for a token pruner.
    A VLM takes in a pruner on initialization and uses prune() to reduce the number of tokens with each layer.
    """
    def __init__(self):
        pass

    def prune(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_scores: torch.Tensor,
        token_types: torch.Tensor,
    ):
        """
        prune() will be called in each decoder layer's forward pass.
        this turns the decoder layer into a function that maps (B, T_1, D) -> (B, T_2, D) where T_2 <= T_1.
        """
        pass