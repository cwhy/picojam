import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from typing import Tuple, List
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.my_datasets import ImageClassification


# Visualize position embeddings and their correlations
def visualize_position_embeddings(pos_network_forward, height_net_params, width_net_params, height, width, embed_dim):
    """Visualize the learned positional embeddings from the networks.
    
    Args:
        height_net_params: Parameters for height embedding network
        width_net_params: Parameters for width embedding network
        height: Image height
        width: Image width
        embed_dim: Embedding dimension
    """
    # Generate normalized coordinate values
    height_coords = jnp.linspace(0, 1, height).reshape(-1, 1)
    width_coords = jnp.linspace(0, 1, width).reshape(-1, 1)
    
    # Compute embeddings for all positions
    height_embeddings = pos_network_forward(height_net_params, height_coords)
    width_embeddings = pos_network_forward(width_net_params, width_coords)
    
    # Create figure with multiple plots
    fig = plt.figure(figsize=(18, 10))
    grid = plt.GridSpec(2, 3, figure=fig)
    
    # 1. Plot height embeddings
    ax1 = fig.add_subplot(grid[0, 0])
    im1 = ax1.imshow(height_embeddings, aspect='auto', cmap='viridis')
    ax1.set_title('Height Network Embeddings')
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Height Position')
    fig.colorbar(im1, ax=ax1)
    
    # 2. Plot width embeddings
    ax2 = fig.add_subplot(grid[0, 1])
    im2 = ax2.imshow(width_embeddings, aspect='auto', cmap='viridis')
    ax2.set_title('Width Network Embeddings')
    ax2.set_xlabel('Embedding Dimension')
    ax2.set_ylabel('Width Position')
    fig.colorbar(im2, ax=ax2)
    
    # 3. Compute and plot height-height correlation matrix
    ax3 = fig.add_subplot(grid[0, 2])
    height_corr = np.corrcoef(height_embeddings)
    im3 = ax3.imshow(height_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_title('Height-Height Correlation')
    ax3.set_xlabel('Height Position')
    ax3.set_ylabel('Height Position')
    fig.colorbar(im3, ax=ax3)
    
    # 4. Compute and plot width-width correlation matrix
    ax4 = fig.add_subplot(grid[1, 0])
    width_corr = np.corrcoef(width_embeddings)
    im4 = ax4.imshow(width_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_title('Width-Width Correlation')
    ax4.set_xlabel('Width Position')
    ax4.set_ylabel('Width Position')
    fig.colorbar(im4, ax=ax4)
    
    # 5. Plot selected embedding dimensions across positions
    ax5 = fig.add_subplot(grid[1, 1])
    
    # Plot a few dimensions for height
    for i in range(0, embed_dim, 4):  # Plot every 4th dimension
        ax5.plot(height_coords, height_embeddings[:, i], 
                 label=f'Height dim {i}', linestyle='-')
    
    # Plot a few dimensions for width
    for i in range(0, embed_dim, 4):  # Plot every 4th dimension
        ax5.plot(width_coords, width_embeddings[:, i], 
                 label=f'Width dim {i}', linestyle='--')
    
    ax5.set_title('Selected Embedding Dimensions')
    ax5.set_xlabel('Normalized Position')
    ax5.set_ylabel('Embedding Value')
    ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax5.grid(True, alpha=0.3)
    
    # 6. Cross-correlation between dimensions
    ax6 = fig.add_subplot(grid[1, 2])
    
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
    fig.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.show()
# Reconstruct and plot images
def reconstruct_images(data, predict_image, height_net_params, width_net_params, mlp_params_matrix, coords, image_indices, height, width):
    """Reconstruct and plot selected images.
    
    Args:
        height_net_params: Parameters for height embedding network
        width_net_params: Parameters for width embedding network
        mlp_params_matrix: Dictionary with parameter matrices for all MLPs
        coords: Coordinate indices (normalized)
        image_indices: Indices of images to reconstruct
        height: Image height
        width: Image width
    """
    num_images = len(image_indices)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
    
    for i, img_idx in enumerate(image_indices):
        # Get original image
        original = data.X[img_idx].reshape(height, width)
        
        # Predict image
        predictions = predict_image(
            (height_net_params, width_net_params), mlp_params_matrix, img_idx, coords)
        reconstructed = predictions.reshape(height, width)
        
        # Plot original
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f"Original #{img_idx}\nLabel: {data.y[img_idx]}")
        axes[0, i].axis('off')
        
        # Plot reconstructed
        axes[1, i].imshow(reconstructed, cmap='gray')
        axes[1, i].set_title(f"Reconstructed #{img_idx}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
# Analyze the learned representations
def analyze_image_representations(data, mlp_params_matrix, num_images=100):
    """Analyze the learned representations by comparing MLP weights.
    
    Args:
        mlp_params_matrix: Dictionary with parameter matrices for all MLPs
        num_images: Number of images to analyze
    """
    # Extract the second layer weights from each MLP
    W2_matrix = mlp_params_matrix['W2'][:num_images]
    
    # Flatten each W2 matrix
    representations = W2_matrix.reshape(W2_matrix.shape[0], -1)
    
    # Get labels
    labels = data.y[:num_images]
    
    # PCA for visualization (simple implementation)
    mean_rep = np.mean(representations, axis=0)
    centered = representations - mean_rep
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    reduced = U[:, :2] * S[:2]
    
    # Plot the representations colored by digit class
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', 
                         alpha=0.8, s=50)
    plt.colorbar(scatter, label='Digit Class')
    plt.title('MLP Representations Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_loss_history(loss_history: List[float]) -> None:
    """Plot the loss history.
    
    Args:
        loss_history: List of loss values during training
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.show()

def visualize_results(data: ImageClassification, 
                     pos_network_forward: callable,
                     predict_image: callable,
                     trained_params: Tuple, 
                     coords: Array, 
                     height: int, 
                     width: int, 
                     embed_dim: int) -> None:
    """Visualize the results of training.
    
    Args:
        data: ImageClassification object containing the dataset
        trained_params: Trained model parameters
        coords: Coordinate indices (normalized)
        height: Image height
        width: Image width
        embed_dim: Dimension of positional embeddings
    """
    # Extract trained parameters
    trained_pos_params, trained_mlp_params_matrix = trained_params
    trained_height_net_params, trained_width_net_params = trained_pos_params
    
    # Visualize position embeddings
    print("Visualizing positional embeddings...")
    visualize_position_embeddings(
        pos_network_forward,
        trained_height_net_params, trained_width_net_params, height, width, embed_dim)
    
    # Reconstruct and plot some images
    test_indices = [0, 10, 20, 30, 40]  # Choose some images to reconstruct
    print("Reconstructing test images...")
    
    # Create a specialized predict function that matches the expected interface
    def predict_fn(pos_params, mlp_params_matrix, img_idx, coords):
        height_net_params, width_net_params = pos_params
        return predict_image(height_net_params, width_net_params, mlp_params_matrix, img_idx, coords)
    
    reconstruct_images(
        data, predict_fn,
        trained_height_net_params, trained_width_net_params, 
        trained_mlp_params_matrix, coords, test_indices, height, width)