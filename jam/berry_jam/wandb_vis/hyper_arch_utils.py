import wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

from shared_utils import log_position_embeddings, log_reconstructed_images






def log_visualizations(epoch, params, coords, flattened_images, height, width, 
                      pos_network_forward, predict_image, data=None, num_train_images=100):
    """Log all visualizations to WandB.
    
    Args:
        epoch: Current epoch number
        params: Model parameters (pos_params, mlp_params_matrix)
        coords: Coordinate indices
        flattened_images: Flattened image data
        height: Image height
        width: Image width
        pos_network_forward: Function for forward pass through position network
        predict_image: Function for predicting an image
        data: Optional dataset object for labels
        num_train_images: Number of training images
    """
    # Unpack parameters
    pos_params, mlp_params_matrix = params
    height_net_params, width_net_params = pos_params
    
    # Extract labels if data is provided
    labels = None
    if data is not None and hasattr(data, 'y'):
        labels = data.y[:num_train_images]
    
    # 1. Log reconstructed images
    vis_indices = np.random.choice(min(num_train_images, flattened_images.shape[0]), 
                                  min(5, flattened_images.shape[0]), 
                                  replace=False)
    
    # Get original images
    original_images = []
    for idx in vis_indices:
        # Make sure we're accessing valid indices
        if idx < flattened_images.shape[0]:
            original = flattened_images[idx].reshape(height, width)
            original_images.append(original)
    
    # Generate reconstructions
    reconstructed_images = []
    for i, idx in enumerate(vis_indices):
        # Make sure we're accessing valid indices
        if idx < num_train_images and i < len(original_images):
            predictions = predict_image(
                height_net_params, width_net_params, 
                mlp_params_matrix, idx, coords
            )
            reconstructed = predictions.reshape(height, width)
            reconstructed_images.append(reconstructed)
    
    # Get labels for these images if available
    vis_labels = None
    if labels is not None:
        vis_labels = [labels[idx] for idx in vis_indices if idx < len(labels)]
    
    # Log images
    if original_images and reconstructed_images:  # Make sure we have images to log
        # Convert to numpy arrays
        orig_np = np.array(original_images)
        recon_np = np.array(reconstructed_images)
        
        # Log to wandb
        log_reconstructed_images(orig_np, recon_np, epoch, vis_labels)
    
    # 2. Log position embeddings visualization
    embed_dim = height_net_params[2].shape[1]  # Extract embedding dimension from params
    log_position_embeddings(
        pos_network_forward,
        height_net_params,
        width_net_params,
        height, width, embed_dim,
        epoch
    )
    
    # 3. Log image representations
    if labels is not None:
        log_image_representations(
            mlp_params_matrix,
            labels,
            epoch,
            min(num_train_images, len(labels))
        )



def log_image_representations(mlp_params_matrix, labels, step, num_images=100):
    """Log PCA visualization of image representations to WandB.
    
    Args:
        mlp_params_matrix: Dictionary with parameter matrices for all MLPs
        labels: Labels for the images
        step: Current training step
        num_images: Number of images to include in visualization
    """
    num_images = min(num_images, len(mlp_params_matrix['W2']), len(labels))
    
    if num_images <= 1:
        return  # Can't do PCA with just one sample
    
    # Extract the second layer weights from each MLP
    W2_matrix = mlp_params_matrix['W2'][:num_images]
    
    # Flatten each W2 matrix
    representations = W2_matrix.reshape(W2_matrix.shape[0], -1)
    
    # Get labels for visualization
    vis_labels = labels[:num_images]
    
    # PCA for visualization (simple implementation)
    mean_rep = np.mean(representations, axis=0)
    centered = representations - mean_rep
    
    # Handle single dimension case or other unexpected shapes
    if centered.shape[0] <= 1 or centered.shape[1] <= 1:
        return  # Can't proceed with PCA
        
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        reduced = U[:, :min(2, U.shape[1])] * S[:min(2, S.shape[0])]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if reduced.shape[1] >= 2:
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=vis_labels, cmap='tab10', 
                                alpha=0.8, s=50)
            plt.colorbar(scatter, label='Digit Class')
        else:
            # Handle 1D case
            scatter = ax.scatter(reduced[:, 0], np.zeros_like(reduced[:, 0]), c=vis_labels, cmap='tab10', 
                                alpha=0.8, s=50)
            plt.colorbar(scatter, label='Digit Class')
        
        ax.set_title('MLP Representations Visualization (PCA)')
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
        wandb.log({"representations": wandb.Image(Image.open(buf))}, step=step)
    except Exception as e:
        print(f"Error generating image representation visualization: {e}")
        return


