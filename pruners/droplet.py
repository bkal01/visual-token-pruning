import torch

from pruners.base_pruner import Pruner

class Droplet(Pruner):
    def __init__(
        self,
        target_layers,
        filtering_ratio,
    ):
        self.target_layers = target_layers
        self.filtering_ratio = filtering_ratio

    def prune_decoder_forward(
        self,
        layer_idx,
        hidden_states,
        token_types,
        **kwargs,
    ):
        """
        Prunes tokens using Droplet. Similar to an embeddings-based approach, but we use the similarity between hidden states to prune.
        """
        T, V = len(token_types), int(token_types.sum())
        if layer_idx not in self.target_layers:
            return hidden_states, torch.ones(T, dtype=torch.bool, device=token_types.device)

        hidden_states = hidden_states.squeeze(0)
        visual_indices, text_indices = token_types.nonzero(as_tuple=True)[0], (~token_types).nonzero(as_tuple=True)[0]
        visual_hidden_states, text_hidden_states = hidden_states[visual_indices], hidden_states[text_indices]

        similarity_scores = torch.matmul(visual_hidden_states, text_hidden_states.T).mean(dim=1)

        amount_to_keep = int(V * (1 - self.filtering_ratio))
        topk_relative = similarity_scores.topk(amount_to_keep).indices.sort().values

        keep_mask = ~token_types.bool()
        keep_mask[visual_indices[topk_relative]] = True
        return hidden_states[keep_mask, :].unsqueeze(0), keep_mask
