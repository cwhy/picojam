import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from jax import Array

def log_embeddings(title: str, embeddings: Array, labels: Array, color_map: str = 'tab10', num: int = 1000) -> Image.Image:
    """Log PCA visualization of embeddings to WandB.
    
    Args:
        embeddings: Array of embeddings 
        labels: Labels for the embeddings
        color_map: Color map to use for the visualization
        num: Number of embeddings to include in visualization
    Returns:
        Image.Image: PCA visualization of embeddings
        None: If the number of embeddings is less than 2
    """
    num = min(num, len(embeddings), len(labels))
    
    if num <= 1:
        raise ValueError("Number of embeddings must be greater than 1, cannot log PCA visualization")
    
    # Extract the embeddings for visualization
    embeddings = embeddings[:num]
    
    # Get labels for visualization
    vis_labels = labels[:num]
    
    # PCA for visualization (simple implementation)
    mean_rep = np.mean(embeddings, axis=0)
    centered = embeddings - mean_rep
    
    # Handle single dimension case or other unexpected shapes
    if centered.shape[0] <= 1 or centered.shape[1] <= 1:
        raise ValueError("Number of embeddings must be greater than 1, cannot log PCA visualization")
        
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        reduced = U[:, :min(2, U.shape[1])] * S[:min(2, S.shape[0])]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if reduced.shape[1] >= 2:
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=vis_labels, cmap=color_map, 
                                alpha=0.8, s=50)
            plt.colorbar(scatter, label='Label')
        else:
            # Handle 1D case
            scatter = ax.scatter(reduced[:, 0], np.zeros_like(reduced[:, 0]), c=vis_labels, cmap=color_map, 
                                alpha=0.8, s=50)
            plt.colorbar(scatter, label='Label')
        
        ax.set_title(f'{title} Embeddings Visualization (PCA)')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2' if reduced.shape[1] >= 2 else 'N/A')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # Log to wandb
        return Image.open(buf)
    except Exception as e:
        raise e

