import time
import jax
import os, sys
import optax
import wandb
import logging
from functools import partial
from jax import Array
import jax.numpy as jnp
from typing import Dict, Any, Iterable, Tuple

from debug_utils import plot_image
from optimizer_bank import get_optimizer
from pos_net import create_coordinate_indices, make_positional_color_map, make_positional_colormap_for_matplotlib
from wandb_vis.vis_methods import log_embeddings


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.random_utils import infinite_safe_keys_from_key
from berries.my_datasets import ImageClassification, load_supervised_image 


logging.basicConfig(level=logging.INFO)

img_dims = (28, 28)  # X dimensions

def init() -> Tuple[Dict[str, Any], Dict[str, Array]]:
    """Initialize all model parameters in a single flat dictionary"""

    config = {
        "dataset_name": "mnist",
        "num_epochs": 10000,
        "embed_dim": 8,  # Total embedding dimension
        "num_train_images": 100000,  # Subset for demonstration
        "batch_size": 2000,
        "learning_rate": 1e-4,
        "random_seed": 42,
        "vis_frequency": 200,  # Visualize every 200 epochs
        "optimizer": "adamw",  # Added optimizer parameter
        "img_dims": img_dims,  # Image dimensions
    }
    key = jax.random.PRNGKey(config["random_seed"])
    embed_dim = config["embed_dim"]
    num_images = config["num_train_images"]
    key_gen = infinite_safe_keys_from_key(key)

    params = {
        'pos_embedding': jax.random.normal(next(key_gen).get(), (*img_dims, embed_dim)) * 0.01,
        'img_embed': jax.random.normal(next(key_gen).get(), (num_images, embed_dim)) * 0.01,
        'bias': jnp.zeros(1, dtype=jnp.float32),
    }
    return config, params, key_gen

def get_positional_embedding(params: Dict[str, Array], coords: Iterable[int]) -> Array:
    """Get embeddings for coordinates"""
    # Extract coordinates
    return params['pos_embedding'][*coords]


def get_image_embeddings(params: Dict[str, Array], img_idx: int) -> Array:
    """Get embeddings for coordinates of a single image"""
    return params['img_embed'][img_idx]

def get_positional_embeddings(params: Dict[str, Array], coords: Array) -> Array:
    """Get embeddings for coordinates"""
    return jax.vmap(get_positional_embedding, in_axes=(None, 0))(params, coords)

def get_pixel_values_1(params: Dict[str, Array], coords: Iterable[int], img_idx: int) -> Array:
    """Get pixel values for a given image index and coordinates of a sigle pixel"""
    pos_embeddings = get_positional_embedding(params, coords)
    img_embeddings = get_image_embeddings(params, img_idx)
    return jax.nn.sigmoid(jnp.dot(pos_embeddings, img_embeddings) + params['bias'])   

def get_image(params: Dict[str, Array], img_idx: int) -> Array:
    """Get the full image for a given image index"""
    coords = create_coordinate_indices(*img_dims)
    pixel_values = jax.vmap(get_pixel_values_1, in_axes=(None, 0, None))(params, coords, img_idx)
    return pixel_values.reshape(img_dims)

def loss_batch_1(params: Dict[str, Array], img_idx: int, target_img: Array) -> Array:
    """Compute the loss for one image"""
    return jnp.mean((get_image(params, img_idx) - target_img) ** 2)

def loss_batch(params: Dict[str, Array], img_idx_batch: Array, target_batch: Array) -> Array:
    """Compute the loss for a batch of images and coordinates"""
    # Compute the loss for each image in the batch
    loss_fn = jax.vmap(loss_batch_1, in_axes=(None, 0, 0))
    return loss_fn(params, img_idx_batch, target_batch)  # type: ignore


@partial(jax.jit, static_argnums=(0, ))
def train_step(optimizer: optax.GradientTransformation, params: Dict[str, Array], opt_state: Any, indices: Array, batch_images: Array) -> Tuple[optax.Params, optax.OptState, Array]:

    def _loss_batch(p):
        return jnp.mean(loss_batch(p, indices, batch_images))

    loss_value, grads = jax.value_and_grad(_loss_batch)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value

def test_train_step_100(params: Dict[str, Array], optimizer: optax.GradientTransformation, X: Array) -> Array:
    opt_state_ = optimizer.init(params)
    indices = jnp.array([0, 1, 2])
    batch_images = X[indices]

    for epoch in range(1000):
        params, opt_state_, loss_value = train_step(optimizer, params, opt_state_, indices, batch_images)
        if epoch % 200 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss_value:.6f}")

    return get_image(params, 0)

    

if __name__ == "__main__":
    use_wandb = True

    # Define configuration dictionary with all parameters
    config, params_, key_gen = init()

    # Debug forward pass
    img_0 = get_image(params_, 0)


    #plot_image(img_0, title="Debug Image", save_path="./tmp0.jpg")

    # Debug batch loss
    loss = loss_batch(params_, jnp.array([0, 1, 2]), jnp.tile(img_0, (3, 1, 1)))
    logging.info(f"Batch Loss: {loss}")

    # Load dataset
    logging.info("Loading dataset...")
    data: ImageClassification = load_supervised_image(config["dataset_name"])
    # Flatten images and normalize pixel values to [0, 1]
    data_X = data.X.reshape(data.n_samples, *img_dims) / 255.0

    optimizer = get_optimizer(config)

    img_100 = test_train_step_100(params_, optimizer, data_X)
    plot_image(img_100, title="Debug Image", save_path="./tmp_100.jpg")

    # Initialize WandB if requested
    if use_wandb:
        logging.info("Initializing WandB...")
        wandb.init(entity="decode-transformer", project="image-reconstruction-embeddings", config=config)

    # Train the model - our refactored version handles parameter initialization internally
    logging.info("Training the model...")

    num_images = config["num_train_images"]
    opt_state_ = optimizer.init(params_)
    start_time = time.perf_counter()
    n_data_ = 0
    loss_history_ = []
    num_batches = num_images // config["batch_size"]
    all_indices = jnp.arange(num_images)

    img_vis_indices = jax.random.choice(next(key_gen).get(), all_indices, shape=(6,), replace=False)

    for epoch in range(config["num_epochs"]):
        epoch_loss_ = 0.0
        key = next(key_gen).get()
        permutation = jax.random.permutation(key, num_images)
        shuffled_images = data_X[permutation]
        shuffled_indices = all_indices[permutation]

        for batch in range(num_batches):
            _start = batch * config["batch_size"]
            _end = _start + config["batch_size"]
            _indices = shuffled_indices[_start:_end]
            params_, opt_state_, _loss_value = train_step(optimizer, params_, opt_state_, _indices, shuffled_images[_start:_end])
            epoch_loss_ += _loss_value

        _avg_loss = epoch_loss_ / num_batches
        loss_history_.append(_avg_loss)

        n_data_ += num_images

        if use_wandb:
            wandb.log({
                "loss": _avg_loss,
                "epoch": epoch,
                "time": time.perf_counter() - start_time,
                "n_data": n_data_
            })

        if epoch % 50 == 0:
            logging.info(f"Epoch {epoch}, Loss: {_avg_loss:.6f}")

        if use_wandb and (epoch % config["vis_frequency"] == 0 or epoch == config["num_epochs"] - 1):
            images = []

            for img_idx in img_vis_indices:
                _img = get_image(params_, img_idx)
                comparison = jnp.hstack([data_X[img_idx], _img])
                images.append(wandb.Image(jax.device_get(comparison), caption=f"Image {img_idx}, Label {data.y[img_idx]}"))

            wandb.log({"image_reconstruction": images})

            wandb.log({"image_embeddings":
                        wandb.Image(
                            log_embeddings(
                                "image",
                                get_image_embeddings(params_, jnp.arange(num_images)),
                                data.y[jnp.arange(num_images)]))})
            _pos_indices = create_coordinate_indices(*img_dims)
            _color_map = make_positional_colormap_for_matplotlib(img_dims)
            labels = jnp.arange(jnp.prod(jnp.array(img_dims)))
            wandb.log({"positional_embeddings":
                        wandb.Image(
                            log_embeddings(
                                "positional",
                                get_positional_embeddings(params_, _pos_indices),
                                labels.tolist(),
                                color_map=_color_map))})
        
    img_final = get_image(params_, 0)
    plot_image(img_final, title="Final Image", save_path="./tmp_final.jpg")

    # Finish WandB run
    if use_wandb:
        wandb.finish()