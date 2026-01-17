import torch

from pruners.base_pruner import Pruner

class TokenEmbeddingsPruner(Pruner):
    def __init__(
        self,
        filtering_ratio,
    ):
        self.filtering_ratio = filtering_ratio


    def prune_embeddings(
        self,
        embeddings: torch.Tensor,
        token_types: torch.Tensor,
    ):
        """
        Prune token embedding inspired by FlashVLM (https://arxiv.org/pdf/2512.20561)
        In my opinion, FlashVLM's approach seems convoluted. They use token embeddings to compute
        similarity scores for pruning, but they do a lot of "massaging" to make it work. So this implementation
        is just loosely inspired by their approach, removing a lot of the extra complexity.
        """
        T, V = len(token_types), int(token_types.sum())
        embeddings = embeddings.squeeze(0) # assume B=1 for now.
        visual_indices, text_indices = token_types.nonzero(as_tuple=True)[0], (~token_types).nonzero(as_tuple=True)[0]
        visual_embeddings, text_embeddings = embeddings[visual_indices], embeddings[text_indices]


        similarity_scores = torch.matmul(visual_embeddings, text_embeddings.T).mean(dim=-1)

        amount_to_keep = int(V * (1 - self.filtering_ratio))
        topk_relative = similarity_scores.topk(amount_to_keep).indices.sort().values


        keep_mask = ~token_types.bool()
        keep_mask[visual_indices[topk_relative]] = True

        return embeddings[keep_mask, :].unsqueeze(0), keep_mask