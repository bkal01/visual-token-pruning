import torch

from abc import ABC
from typing import List

class Pruner(ABC):
    """
    Base class for a token pruner.
    A VLM takes in a pruner on initialization and uses prune() to reduce the number of tokens in a layer.
    """
    def __init__(self):
        pass

    def prune_vision_forward(
        self,
    ):
        """
        prune_vision_forward() will be called in the vision encoder's forward pass.
        """

    def prune_embeddings(
        self,
        embeddings: torch.Tensor,
        token_types: torch.Tensor,
        **kwargs,
    ):
        """
        prune_embeddings() will be called prior to decoder layers. 
        """
        T = len(token_types)
        return embeddings, torch.ones(T, dtype=torch.bool, device=token_types.device)


    def prune_decoder_forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        token_types: torch.Tensor,
        **kwargs,
    ):
        """
        prune_decoder_forward() will be called after a decoder layer's forward pass.
        this turns the decoder layer into a function that maps (B, T + V_1, D) -> (B, T + V_2, D) where V_2 <= V_1.
        """
        T = len(token_types)
        return hidden_states, torch.ones(T, dtype=torch.bool, device=token_types.device)