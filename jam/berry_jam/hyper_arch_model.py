import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import optax
from typing import Tuple, Dict, List, Any
from wandb_vis.hyper_arch_utils import log_visualizations
from wandb_vis.shared_utils import log_epoch_metrics


def create_coordinate_indices(height: int, width: int) -> Array:
    """Create coordinate indices for an image.
    
    Args:
        height: Image height
        width: Image width
    
    Returns:
        Array of shape (height*width, 2) with (y, x) coordinates
    """
    y_indices = jnp.arange(height)
    x_indices = jnp.arange(width)
    
    # Create meshgrid
    y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')
    
    # Reshape to (height*width, 2)
    coords = jnp.stack([y_grid.flatten(), x_grid.flatten()], axis=1)
    
    # Normalize to [0, 1]
    coords = coords.astype(jnp.float32)
    coords = coords / jnp.array([height - 1, width - 1])
    
    return coords

def init_pos_network_params(key: Array, hidden_dim: int = 32, embed_dim: int = 16) -> Tuple:
    """Initialize parameters for a positional embedding network.
    
    Args:
        key: Random key for initialization
        hidden_dim: Hidden layer dimension
        embed_dim: Output embedding dimension
        
    Returns:
        Tuple of (W1, b1, W2, b2)
    """
    key1, key2 = jax.random.split(key)
    
    # Initialize weights for two layers (input is a single coordinate value)
    W1 = jax.random.normal(key1, (1, hidden_dim)) * jnp.sqrt(2.0) * 5 
    b1 = jnp.zeros((hidden_dim,))

    W2 = jax.random.normal(key2, (hidden_dim, embed_dim)) * jnp.sqrt(2.0 / hidden_dim)
    b2 = jnp.zeros((embed_dim,))
    
    return (W1, b1, W2, b2)

def pos_network_forward(params: Tuple, x: Array) -> Array:
    """Forward pass through the positional embedding network.
    
    Args:
        params: Tuple of (W1, b1, W2, b2)
        x: Input array of shape (batch_size, 1) with coordinate values
        
    Returns:
        Output array of shape (batch_size, embed_dim)
    """
    W1, b1, W2, b2 = params
    
    # First layer with sine activation for better positional encoding
    h = jnp.dot(x, W1) + b1
    h = jnp.sin(h)  # Sine activation helps with smooth positional representations

    # Second layer with linear activation
    y = jnp.dot(h, W2) + b2
    
    return y

def init_mlp_params(key: Array, input_dim: int, hidden_dim: int = 64) -> Tuple:
    """Initialize parameters for a small MLP.
    
    Args:
        key: Random key for initialization
        input_dim: Input dimension (2*embed_dim)
        hidden_dim: Hidden layer dimension
        
    Returns:
        Tuple of (W1, b1, W2, b2)
    """
    key1, key2 = jax.random.split(key)
    
    # Initialize weights for two layers
    W1 = jax.random.normal(key1, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim)
    b1 = jnp.zeros((hidden_dim,))
    
    W2 = jax.random.normal(key2, (hidden_dim, 1)) * jnp.sqrt(2.0 / hidden_dim)
    b2 = jnp.zeros((1,))
    
    return (W1, b1, W2, b2)

def init_all_mlp_params(key: Array, num_images: int, input_dim: int, hidden_dim: int = 64) -> Dict[str, Array]:
    """Initialize parameters for all image MLPs at once.
    
    Args:
        key: Random key for initialization
        num_images: Number of images to create parameters for
        input_dim: Input dimension (2*embed_dim)
        hidden_dim: Hidden layer dimension
        
    Returns:
        Dictionary with parameter matrices for all MLPs
        {
            'W1': Array of shape (num_images, input_dim, hidden_dim),
            'b1': Array of shape (num_images, hidden_dim),
            'W2': Array of shape (num_images, hidden_dim, 1),
            'b2': Array of shape (num_images, 1)
        }
    """
    # Generate keys for all images
    keys = jax.random.split(key, num_images)
    
    # Initialize parameters for each image
    all_W1 = []
    all_b1 = []
    all_W2 = []
    all_b2 = []
    
    for i in range(num_images):
        W1, b1, W2, b2 = init_mlp_params(keys[i], input_dim, hidden_dim)
        all_W1.append(W1)
        all_b1.append(b1)
        all_W2.append(W2)
        all_b2.append(b2)
    
    # Stack parameters into matrices
    mlp_params = {
        'W1': jnp.stack(all_W1),
        'b1': jnp.stack(all_b1),
        'W2': jnp.stack(all_W2),
        'b2': jnp.stack(all_b2)
    }
    
    return mlp_params

def get_positional_embeddings(height_net_params: Tuple, width_net_params: Tuple, coords: Array) -> Array:
    """Get embeddings for coordinates using positional networks.
    
    Args:
        height_net_params: Parameters for height embedding network
        width_net_params: Parameters for width embedding network
        coords: Array of shape (batch_size, 2) with normalized (y, x) coordinates
    
    Returns:
        Array of shape (batch_size, 2*embed_dim) with concatenated embeddings
    """
    # Extract normalized y and x coordinates
    y_coords = coords[:, 0:1]  # Keep the dimension for matrix operations
    x_coords = coords[:, 1:2]
    
    # Get embeddings using the networks
    y_embeddings = pos_network_forward(height_net_params, y_coords)
    x_embeddings = pos_network_forward(width_net_params, x_coords)
    
    # Concatenate embeddings
    return jnp.concatenate([y_embeddings, x_embeddings], axis=1)

def predict_image(height_net_params: Tuple, width_net_params: Tuple, 
                 mlp_param_matrix: Dict[str, Array], img_idx: int, coords: Array) -> Array:
    """Predict pixel values for a single image using its parameters.
    
    Args:
        height_net_params: Parameters for height embedding network
        width_net_params: Parameters for width embedding network
        mlp_param_matrix: Dictionary with parameter matrices for all MLPs
        img_idx: Index of the image to predict
        coords: Coordinate indices (normalized)
        
    Returns:
        Predicted pixel values for the image
    """
    # Get positional embeddings
    pos_embeddings = get_positional_embeddings(height_net_params, width_net_params, coords)
    
    # Extract parameters for this image
    W1, b1 = mlp_param_matrix['W1'][img_idx], mlp_param_matrix['b1'][img_idx]
    W2, b2 = mlp_param_matrix['W2'][img_idx], mlp_param_matrix['b2'][img_idx]
    
    # First layer with ReLU activation
    h = jnp.dot(pos_embeddings, W1) + b1
    h = jax.nn.relu(h)
    
    # Second layer with sigmoid activation for pixel values
    y = jnp.dot(h, W2) + b2
    y = jax.nn.sigmoid(y)
        
    return y

def compute_single_image_loss(height_net_params: Tuple, width_net_params: Tuple, 
                             W1: Array, b1: Array, W2: Array, b2: Array, 
                             coords: Array, target_image: Array) -> Array:
    """Compute MSE loss for a single image.
    
    Args:
        height_net_params: Parameters for height embedding network
        width_net_params: Parameters for width embedding network
        W1, b1, W2, b2: MLP parameters for this image
        coords: Coordinate indices (normalized)
        target_image: Target image pixels (flattened)
    
    Returns:
        MSE loss
    """
    # Get positional embeddings
    pos_embeddings = get_positional_embeddings(height_net_params, width_net_params, coords)
    
    # First layer with ReLU activation
    h = jnp.dot(pos_embeddings, W1) + b1
    h = jax.nn.relu(h)
    
    # Second layer with sigmoid activation for pixel values
    y = jnp.dot(h, W2) + b2
    y = jax.nn.sigmoid(y)
    
    # Calculate mean squared error
    return jnp.mean((y.flatten() - target_image) ** 2)

def compute_batch_loss(height_net_params: Tuple, width_net_params: Tuple, 
                      mlp_params_matrix: Dict[str, Array], 
                      coords: Array, target_images_batch: Array) -> Array:
    """Compute average MSE loss for a batch of images.
    
    Args:
        height_net_params: Parameters for height embedding network
        width_net_params: Parameters for width embedding network
        mlp_params_matrix: Dictionary with parameter matrices for all MLPs
        coords: Coordinate indices (normalized)
        target_images_batch: Batch of target image pixels
    
    Returns:
        Average MSE loss over the batch
    """
    # Get positional embeddings (same for all images in batch)
    pos_embeddings = get_positional_embeddings(height_net_params, width_net_params, coords)
    
    # Define a function that computes loss for a single image given its MLP params and target
    def single_image_loss(W1, b1, W2, b2, target_image):
        # First layer with ReLU activation
        h = jnp.dot(pos_embeddings, W1) + b1
        h = jax.nn.relu(h)
        
        # Second layer with sigmoid activation for pixel values
        y = jnp.dot(h, W2) + b2
        y = jax.nn.sigmoid(y)
        
        # Calculate mean squared error
        return jnp.mean((y.flatten() - target_image) ** 2)
    
    # Map the loss function over all parameters and targets
    losses = jax.vmap(single_image_loss)(
        W1=mlp_params_matrix['W1'],
        b1=mlp_params_matrix['b1'],
        W2=mlp_params_matrix['W2'],
        b2=mlp_params_matrix['b2'],
        target_image=target_images_batch
    )
    
    # Return average loss
    return jnp.mean(losses)

def create_train_step(optimizer):
    """Create a JIT-compiled training step function.
    
    Args:
        optimizer: Optax optimizer
        
    Returns:
        JIT-compiled training step function
    """
    @jax.jit
    def train_step(params: Tuple, opt_state: Any, coords: Array, 
                  target_images_batch: Array, batch_idx: int) -> Tuple:
        """Perform a single training step for a batch of images.
        
        Args:
            params: Tuple of (pos_params, mlp_params_matrix)
            opt_state: Optimizer state
            coords: Coordinate indices (normalized)
            target_images_batch: Batch of target image pixels
            batch_idx: Integer representing the batch index
        
        Returns:
            Updated parameters, optimizer state, and loss value
        """
        # Unpack parameters
        pos_params, mlp_params_matrix = params
        height_net_params, width_net_params = pos_params
        
        # Define the loss function 
        def loss_fn(p):
            pos_p, mlp_p_matrix = p
            height_p, width_p = pos_p
            return compute_batch_loss(height_p, width_p, mlp_p_matrix, coords, target_images_batch)
        
        # Compute gradients
        loss_value, grads = jax.value_and_grad(loss_fn)(params)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss_value
    
    return train_step

def train_model(params: Tuple, coords: Array, flattened_images: Array, 
               optimizer: optax.GradientTransformation,
               num_epochs: int = 200, batch_size: int = 10, 
               num_train_images: int = 100, 
               use_wandb: bool = False,
               vis_frequency: int = 50,
               height: int = 28, width: int = 28,
               data=None,
               pos_network_forward=None,
               predict_image=None) -> Tuple[Tuple, List[float]]:
    """Train the model on the dataset with comprehensive WandB visualization.
    
    Args:
        params: Initial model parameters (pos_params, mlp_params_matrix)
        coords: Coordinate indices (normalized)
        flattened_images: Flattened image data
        optimizer: Optax optimizer
        num_epochs: Number of training epochs
        batch_size: Number of images to process in each batch
        num_train_images: Total number of training images
        use_wandb: Whether to use WandB visualization
        vis_frequency: How often to generate visualizations
        height, width: Image dimensions
        data: Optional dataset object for labels
        pos_network_forward: Function for position network forward pass
        predict_image: Function for predicting images
        
    Returns:
        Trained parameters and loss history
    """
    # Create the training step function
    train_step_fn = create_train_step(optimizer)
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    loss_history = []
    
    # Number of batches
    num_batches = num_train_images // batch_size
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Shuffle the dataset
        permutation = np.random.permutation(num_train_images)
        shuffled_images = flattened_images[permutation]
        
        # Process each batch
        for batch in range(num_batches):
            # Get batch data
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_images = shuffled_images[start_idx:end_idx]
            
            # Train step
            params, opt_state, loss = train_step_fn(
                params, opt_state, coords, batch_images, batch)
            
            epoch_loss += loss
        
        # Average loss for the epoch
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Log loss to WandB
        if use_wandb:
            log_epoch_metrics(epoch, avg_loss)
        
        # Log visualizations periodically
        if use_wandb and (epoch % vis_frequency == 0 or epoch == num_epochs - 1):
            log_visualizations(
                epoch=epoch,
                params=params, 
                coords=coords, 
                flattened_images=flattened_images, 
                height=height, 
                width=width, 
                pos_network_forward=pos_network_forward,
                predict_image=predict_image,
                data=data, 
                num_train_images=num_train_images
            )
                
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    return params, loss_history