from doctest import debug
from typing import Dict
import jax
import os, sys
import optax
import wandb
from embed_model import (
    debug_jax,
    train_model
)
from jam.berries.random_utils import infinite_safe_keys_from_key
from pos_net import create_coordinate_indices
import logging

logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.my_datasets import ImageClassification, load_supervised_image 

def init_embed_params(key: jax.Array, num_images: int, img_dim: int,
                     embed_dim: int = 16) -> Dict[str, jax.Array]:
    """Initialize all model parameters in a single flat dictionary"""
    key_gen = infinite_safe_keys_from_key(key)

    params = {
        'pos_embedding': jax.random.normal(next(key_gen).get(), (img_dim, embed_dim * 2)) * 0.01,
        'img_embed': jax.random.normal(next(key_gen).get(), (num_images, embed_dim * 2)) * 0.01,
    }
    return params

@jax.jit
def get_positional_embeddings(params: Dict[str, jax.Array], coords: jax.Array) -> jax.Array:
    """Get embeddings for coordinates using positional networks"""
    # Extract coordinates
    y_coords, x_coords  = coords[:, 0:1], coords[:, 1:2]
    return params['pos_embedding']

@jax.jit
def get_image_embeddings(params: Dict[str, jax.Array], target: jax.Array) -> jax.Array:
    """Get embeddings for coordinates using positional networks"""
    return params['img_embed']

@jax.jit
def get_pixel_values(params: Dict[str, jax.Array], pos_embedding: jax.Array, target: jax.Array) -> jax.Array:
    """Get embeddings for coordinates using positional networks"""
    return params['img_embed'] * pos_embedding

@jax.jit
def predict_image(params: Dict[str, jax.Array], img_idx: int, coords: jax.Array, target: jax.Array) -> jax.Array:
    """Predict pixel values for a single image"""
    pos_embeddings = get_positional_embeddings(params, coords)
    img_embeddings = get_image_embeddings(params, target)
    return get_pixel_values(params, pos_embeddings, img_embeddings)


if __name__ == "__main__":
    use_wandb = True

    # Define configuration dictionary with all parameters
    config = {
        "dataset_name": "mnist",
        "num_epochs": 10000,
        "embed_dim": 4,  # Total embedding dimension
        "num_train_images": 100000,  # Subset for demonstration
        "batch_size": 2000,
        "learning_rate": 1e-4,
        "random_seed": 42,
        "vis_frequency": 200,  # Visualize every 200 epochs
        "optimizer": "adam"  # Added optimizer parameter
    }

    # Initialize random key
    key = jax.random.PRNGKey(config["random_seed"])
    params = init_embed_params(key, config["num_train_images"], config["height"] * config["width"])

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
        wandb.init(entity="decode-transformer", project="image-reconstruction-embeddings")
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