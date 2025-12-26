import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_attn_heatmap(attn_scores, save_path=None):
    """
    plots a heatmap of the attention scores for a single layer, shape (T, T)
    """
    # apply lower triangular mask
    masked_attn_scores = np.tril(attn_scores.cpu().numpy())
    masked_attn_scores[masked_attn_scores == 0] = np.nan

    plt.figure(figsize=(10, 10))
    plt.imshow(masked_attn_scores, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("Attention Heatmap")
    if save_path:
        plt.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_last_attn_intensity_distribution(attn_scores, save_path=None):
    """
    plot "last attention", the attention scores during prefill
    assume scores are of shape (T, T). average over head dimension
    """
    last_scores = attn_scores[-1, :].cpu().numpy()
    
    plt.figure(figsize=(5, 5))
    plt.hist(last_scores, bins=50, density=True, edgecolor="none")
    plt.yscale("log")
    plt.xlabel("Attention Score")
    plt.ylabel("Proportion")
    plt.title("Last Attention Distribution")
    
    if save_path:
        plt.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_attn_visual_tokens_distribution(attn_scores, token_types, save_path=None):
    visual_mask = token_types == 1
    visual_indices = visual_mask.nonzero(as_tuple=True)[0]
    V = len(visual_indices)
    
    last_visual_idx = visual_indices[-1].item()
    indices = torch.arange(len(token_types), device=token_types.device)
    text_after_mask = (token_types == 0) & (indices > last_visual_idx) & (indices < len(token_types) - 1)
    text_after_indices = text_after_mask.nonzero(as_tuple=True)[0]
    
    visual_to_visual = attn_scores[visual_indices][:, visual_indices]
    visual_from_visual = visual_to_visual.mean(dim=0).cpu().numpy()
    
    text_to_visual = attn_scores[text_after_indices][:, visual_indices]
    visual_from_text = text_to_visual.mean(dim=0).cpu().numpy()
    
    visual_from_last = attn_scores[-1, visual_indices].cpu().numpy()
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(range(V), visual_from_visual)
    axes[0].set_xlabel("Visual Token Index")
    axes[0].set_ylabel("Avg Attention Score")
    axes[0].set_title("From Visual Tokens")
    
    axes[1].bar(range(V), visual_from_text)
    axes[1].set_xlabel("Visual Token Index")
    axes[1].set_ylabel("Avg Attention Score")
    axes[1].set_title("From Text Tokens (after)")
    
    axes[2].bar(range(V), visual_from_last)
    axes[2].set_xlabel("Visual Token Index")
    axes[2].set_ylabel("Attention Score")
    axes[2].set_title("From Last Token")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
