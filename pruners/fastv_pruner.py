import torch

from pruners.base_pruner import Pruner

class FastVPruner(Pruner):
    def __init__(
        self,
        layer_threshold,
        filtering_ratio,
    ):
        """
        layer_threshold: if the layer # is less than this, no pruning is done
        filtering_ratio: float between 0 and 1, the fraction of tokens to remove at each layer
        """
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
        Prunes tokens using FastV: at each layer, keep the tokens with the highest average attention score received from all other tokens.
        attention_scores is a tensor of shape (T, T), where T is the sequence length.
        token_types is a tensor of shape (T,) with value 0 for text tokens and 1 for visual tokens.
        """
        T = len(token_types)
        V = int(token_types.sum())
        if layer_idx < self.layer_threshold:
            return hidden_states, token_types, torch.ones(T, dtype=torch.bool, device=token_types.device), torch.arange(V, device=token_types.device)

        # we are going to:
        # - ignore how much each token attends to itself (was in the phrasing of the FastV paper, not sure if this is correct).
        # - find indices of topk visual tokens by average attention score received from all other tokens.
        # - build a keep mask that is True for the tokens to keep (text tokens and topk visual tokens) and False otherwise.
        # - index into hidden states with the keep mask.
        attention_scores *= (1 - torch.eye(T, device=attention_scores.device))
        visual_token_scores = attention_scores[:, token_types == 1].mean(dim=0)

        visual_indices = token_types.nonzero(as_tuple=True)[0]

        amount_to_keep = int(V * (1 - self.filtering_ratio))
        topk_relative = visual_token_scores.topk(amount_to_keep).indices.sort().values

        keep_mask = ~token_types.bool()
        keep_mask[visual_indices[topk_relative]] = True
        return hidden_states[:, keep_mask, :], token_types[keep_mask], keep_mask, topk_relative
