import io
import matplotlib.pyplot as plt
import numpy as np

def plot_attn_heatmap(attn_scores, save_path=None):
    """
    plots a heatmap of the attention scores, shape (T, T)
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


