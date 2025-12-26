import io
import matplotlib.pyplot as plt
import numpy as np

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
