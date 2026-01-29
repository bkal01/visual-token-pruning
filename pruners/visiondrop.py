import torch

from pruners.base_pruner import Pruner

class VisionDropPruner(Pruner):
    def __init__(
        self,
        vision_target_layers,
        llm_target_layers,
        filtering_ratio,
    ):
        """
        vision_target_layers: list of layers in the vision encoder to prune after.
        llm_target_layers: list of layers in the LLM to prune after.
        filtering_ratio: float between 0 and 1, the fraction of tokens to remove at each target layer
        """
        self.vision_target_layers = vision_target_layers
        self.llm_target_layers = llm_target_layers
        self.filtering_ratio = filtering_ratio

    def prune_vision_forward(
        self,
        layer_idx,
        hidden_states,
        **kwargs,
    ):
        """
        Prunes tokens using VisionDrop (https://arxiv.org/pdf/2506.22283).
        As a modification, we average attention scores across groups of patches of size spatial_merge_size^2.
        Then when pruning, we prune entire groups of patches at once.
        """
        attention_scores = kwargs["attention_scores"]
        spatial_merge_size = kwargs["spatial_merge_size"]
        group_size = spatial_merge_size**2

        # P is the number of patches in the VLM.
        P = attention_scores.shape[0]
        if P % group_size != 0:
            raise ValueError(f"P must be divisible by spatial_merge_size**2, but P={P} and spatial_merge_size={spatial_merge_size}")

        if layer_idx not in self.vision_target_layers:
            return hidden_states, torch.ones(P, dtype=torch.bool, device=attention_scores.device)


        visual_token_scores = attention_scores.mean(dim=0)
        per_group_scores = visual_token_scores.view((P // group_size, group_size)).mean(dim=1)

        amount_to_keep = int(P // group_size * (1 - self.filtering_ratio))
        topk_indices = per_group_scores.topk(amount_to_keep).indices

        keep_mask = torch.zeros(P, dtype=torch.bool, device=attention_scores.device)
        offsets = torch.arange(group_size, device=attention_scores.device)
        expanded_indices = (topk_indices.unsqueeze(1) * group_size + offsets).flatten()
        keep_mask[expanded_indices] = True
        return hidden_states[keep_mask, :], keep_mask