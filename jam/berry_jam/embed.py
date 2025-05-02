from doctest import debug
from functools import partial
import jax
import os, sys
import optax
import wandb
from embed_model import (
    debug_jax,
    train_model
)
from pos_net import create_coordinate_indices
import logging

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.my_datasets import ImageClassification, load_supervised_image 

if __name__ == "__main__":
    # class ImageClassification(NamedTuple):
    #     n_samples: int
    #     d_x: tuple[int, int]
    #     d_y: int
    #     n_channels: int
    #     X: Array
    #     y: Array
    #     X_test: Array
    #     y_test: Array

    use_wandb = True

    # Define configuration dictionary with all parameters
    config = {
        "dataset_name": "mnist",
        "num_epochs": 10000,
        "embed_dim": 4,  # Total embedding dimension
        "hidden_dim": 16,
        "qk_dim": 16,
        "num_train_images": 100000,  # Subset for demonstration
        "batch_size": 2000,
        "learning_rate": 1e-4,
        "random_seed": 42,
        "vis_frequency": 200,  # Visualize every 200 epochs
        "optimizer": "adamw"  # Added optimizer parameter
    }

    # Initialize random key
    key = jax.random.PRNGKey(config["random_seed"])
    out = debug_jax(
        key,
        784,
        config["num_train_images"],
        config["hidden_dim"],
        config["embed_dim"]
    )
    print(out.shape)

    # Load dataset
    logging.info("Loading dataset...")
    data: ImageClassification = load_supervised_image(config["dataset_name"])
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


    # Flatten images and normalize pixel values to [0, 1]
    flattened_images = data.X.reshape(data.n_samples, -1) / 255.0
    if config["optimizer"] == "adam":
        optimizer = optax.adam(config["learning_rate"])
    elif config["optimizer"] == "adamw":
        optimizer = optax.adamw(config["learning_rate"])
    elif config["optimizer"] == "rmsprop":
        optimizer = optax.rmsprop(config["learning_rate"])
    elif config["optimizer"] == "muon":
        optimizer = optax.contrib.muon(config["learning_rate"])
    else:
        optimizer = optax.sgd(config["learning_rate"])
    
    

    # Train the model - our refactored version handles parameter initialization internally
    logging.info("Training the model...")
    trained_params, loss_history = train_model(
        key=key,
        coords=coords,
        flattened_images=flattened_images,
        num_images=config["num_train_images"],
        hidden_dim=config["hidden_dim"],
        embed_dim=config["embed_dim"],
        qk_dim=config["qk_dim"],
        optimizer=optimizer,
        use_wandb=use_wandb,
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        vis_frequency=config["vis_frequency"],
        height=config["height"],
        width=config["width"],
        data=data  # Pass data for labels
    )

    # Finish WandB run
    if use_wandb:
        wandb.finish()