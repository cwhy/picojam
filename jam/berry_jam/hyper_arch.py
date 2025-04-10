
import jax
from jax import Array
import optax
from typing import Protocol, Tuple
import os, sys
import wandb
from hyper_arch_model import create_coordinate_indices, init_all_mlp_params, init_pos_network_params, pos_network_forward, predict_image, train_model
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.my_datasets import load_supervised_image 
if __name__ == "__main__":
    class ImageClassificationProtocol(Protocol):
        @property
        def n_samples(self) -> int: ...
        
        @property
        def d_x(self) -> Tuple[int, int]: ...
        
        @property
        def d_y(self) -> int: ...
        
        @property
        def n_channels(self) -> int: ...
        
        @property
        def X(self) -> Array: ...
        
        @property
        def y(self) -> Array: ...
        
        @property
        def X_test(self) -> Array: ...
        
        @property
        def y_test(self) -> Array: ...


    use_wandb = True

    # Define configuration dictionary with all parameters
    config = {
        "dataset_name": "mnist",
        "num_epochs": 2000,
        "embed_dim": 4,
        "hidden_dim": 8,
        "pos_hidden_dim": 1000,
        "num_train_images": 50,  # Subset for demonstration
        "batch_size": 50,
        "learning_rate": 1e-4,
        "random_seed": 42,
        "vis_frequency": 100  # Visualize every 100 epochs
    }

    # Load dataset
    logging.info("Loading dataset...")
    data: ImageClassificationProtocol = load_supervised_image(config["dataset_name"])
    height, width = data.d_x

    # Add image dimensions to config
    config["height"] = height
    config["width"] = width

    # Initialize WandB if requested
    if use_wandb:
        logging.info("Initializing WandB...")
        wandb.init(entity="decode-transformer", project="image-reconstruction-hypernetwork")
        wandb.config.update(config)

    # Create coordinate indices (normalized)
    coords = create_coordinate_indices(config["height"], config["width"])

    # Initialize models
    key = jax.random.PRNGKey(config["random_seed"])
    height_key, width_key, mlp_key = jax.random.split(key, 3)

    # Initialize positional embedding networks
    height_net_params = init_pos_network_params(
        height_key, 
        hidden_dim=config["pos_hidden_dim"], 
        embed_dim=config["embed_dim"]
    )
    width_net_params = init_pos_network_params(
        width_key, 
        hidden_dim=config["pos_hidden_dim"], 
        embed_dim=config["embed_dim"]
    )

    # Combined positional parameters
    pos_params = (height_net_params, width_net_params)

    # Initialize MLP parameters for all images as a matrix
    mlp_params_matrix = init_all_mlp_params(
        mlp_key, 
        config["num_train_images"], 
        input_dim=2*config["embed_dim"], 
        hidden_dim=config["hidden_dim"]
    )

    # Combine all parameters
    params = (pos_params, mlp_params_matrix)

    # Create optimizer
    optimizer = optax.adam(config["learning_rate"])

    # Flatten images and normalize pixel values to [0, 1]
    flattened_images = data.X.reshape(data.n_samples, -1) / 255.0

    # Train the model
    logging.info("Training the model...")
    trained_params, loss_history = train_model(
        params, 
        coords, 
        flattened_images, 
        optimizer,
        use_wandb=use_wandb,

        num_epochs=config["num_epochs"], 
        batch_size=config["batch_size"], 
        num_train_images=config["num_train_images"],
        vis_frequency=config["vis_frequency"],
        height=config["height"],
        width=config["width"],

        data=data,  # Pass data for labels
        pos_network_forward=pos_network_forward,  # Pass the necessary functions
        predict_image=predict_image
    )

    # Finish WandB run
    if use_wandb:
        wandb.finish()