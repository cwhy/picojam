import wandb
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
import io


def log_epoch_metrics(epoch: int, avg_loss: float, n_data: int, seconds: int):
    """Log metrics for a single epoch.
    
    Args:
        epoch: Current epoch number
        avg_loss: Average loss for the epoch
    """
    # print(f"Epoch {epoch}: Average Loss: {avg_loss:.4f}, Data Points: {n_data}, Time: {seconds} seconds")
    wandb.log({"loss-epoch": float(avg_loss)}, step=epoch)
    # wandb.log({"loss-token": float(avg_loss)}, step=n_data)
    # wandb.log({"traing-speed": float(n_data)}, step=seconds)

def log_position_embeddings(pos_network_forward, height_net_params, width_net_params, 
                           height, width, embed_dim, step):
    """Log all the position embedding visualizations to WandB.
    
    Args:
        pos_network_forward: Function to generate embeddings
        height_net_params: Parameters for height embedding network
        width_net_params: Parameters for width embedding network
        height: Image height
        width: Image width
        embed_dim: Embedding dimension
        step: Current training step
    """
    
    # Generate normalized coordinate values
    height_coords = jnp.linspace(0, 1, height).reshape(-1, 1)
    width_coords = jnp.linspace(0, 1, width).reshape(-1, 1)
    
    # Compute embeddings for all positions
    height_embeddings = np.array(pos_network_forward(height_net_params, height_coords))
    width_embeddings = np.array(pos_network_forward(width_net_params, width_coords))
    
    # 1. Plot height and width embeddings
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    im1 = ax1.imshow(height_embeddings, aspect='auto', cmap='viridis')
    ax1.set_title('Height Network Embeddings')
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Height Position')
    fig1.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(width_embeddings, aspect='auto', cmap='viridis')
    ax2.set_title('Width Network Embeddings')
    ax2.set_xlabel('Embedding Dimension')
    ax2.set_ylabel('Width Position')
    fig1.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # Save figure to buffer
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    plt.close(fig1)
    buf1.seek(0)
    
    # 2. Plot correlation matrices
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compute correlation matrices
    height_corr = np.corrcoef(height_embeddings)
    width_corr = np.corrcoef(width_embeddings)
    
    im3 = ax3.imshow(height_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_title('Height-Height Correlation')
    ax3.set_xlabel('Height Position')
    ax3.set_ylabel('Height Position')
    fig2.colorbar(im3, ax=ax3)
    
    im4 = ax4.imshow(width_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_title('Width-Width Correlation')
    ax4.set_xlabel('Width Position')
    ax4.set_ylabel('Width Position')
    fig2.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    
    # Save figure to buffer
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    plt.close(fig2)
    buf2.seek(0)
    
    # 3. Plot selected embedding dimensions
    fig3, ax5 = plt.subplots(figsize=(10, 6))
    
    # Plot a few dimensions for height
    for i in range(0, embed_dim, max(1, embed_dim // 4)):  # Plot a few dimensions
        ax5.plot(np.array(height_coords).flatten(), height_embeddings[:, i], 
                 label=f'Height dim {i}', linestyle='-')
    
    # Plot a few dimensions for width
    for i in range(0, embed_dim, max(1, embed_dim // 4)):  # Plot a few dimensions
        ax5.plot(np.array(width_coords).flatten(), width_embeddings[:, i], 
                 label=f'Width dim {i}', linestyle='--')
    
    ax5.set_title('Selected Embedding Dimensions')
    ax5.set_xlabel('Normalized Position')
    ax5.set_ylabel('Embedding Value')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure to buffer
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png')
    plt.close(fig3)
    buf3.seek(0)
    
    # 4. Cross-correlation between dimensions
    fig4, ax6 = plt.subplots(figsize=(8, 7))
    
    # Compute correlation between embedding dimensions
    dim_corr = np.zeros((embed_dim, embed_dim))
    for i in range(embed_dim):
        for j in range(embed_dim):
            # Use middle positions from both dimensions for correlation
            h_mid = height_embeddings[height // 2, i]
            w_vals = width_embeddings[:, j]
            # Correlation between one height position and all width positions
            corr = np.corrcoef(h_mid * np.ones_like(w_vals), w_vals)[0, 1]
            dim_corr[i, j] = corr
    
    im6 = ax6.imshow(dim_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax6.set_title('Height-Width Dimension Correlation')
    ax6.set_xlabel('Width Dimension')
    ax6.set_ylabel('Height Dimension')
    fig4.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    
    # Save figure to buffer
    buf4 = io.BytesIO()
    fig4.savefig(buf4, format='png')
    plt.close(fig4)
    buf4.seek(0)
    
    # Log all images to wandb
    wandb.log({
        "embeddings/height_width": wandb.Image(Image.open(buf1)),
        "embeddings/correlation_matrices": wandb.Image(Image.open(buf2)),
        "embeddings/dimension_profiles": wandb.Image(Image.open(buf3)),
        "embeddings/cross_correlation": wandb.Image(Image.open(buf4))
    }, step=step)

def log_reconstructed_images(original_images, reconstructed_images, step, labels=None):
    """Log original and reconstructed images to WandB.
    
    Args:
        original_images: Batch of original images (normalized to [0, 1])
        reconstructed_images: Batch of reconstructed images (normalized to [0, 1])
        step: Current training step
        labels: Optional labels for the images
    """
    # Convert images to wandb format
    wandb_images = []
    
    for i in range(min(len(original_images), len(reconstructed_images))):
        # Convert from [0, 1] to [0, 255]
        orig = (original_images[i] * 255).astype(np.uint8)
        recon = (reconstructed_images[i] * 255).astype(np.uint8)
        
        # Create side-by-side comparison
        comparison = np.hstack([orig, recon])
        
        # Add caption with label if available
        if labels is not None and i < len(labels):
            caption = f"Original vs Reconstructed (Image {i}, Label: {labels[i]})"
        else:
            caption = f"Original vs Reconstructed (Image {i})"
            
        wandb_images.append(wandb.Image(comparison, caption=caption))
    
    wandb.log({"reconstructions": wandb_images}, step=step)