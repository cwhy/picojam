import time
import jax
import os, sys
import optax
import wandb
import logging
import math
from functools import partial
from jax import Array
import jax.numpy as jnp
from typing import Dict, Any, Tuple

from debug_utils import plot_image
from optimizer_bank import get_optimizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.random_utils import infinite_safe_keys_from_key
from berries.my_datasets import ImageClassification, load_supervised_image 


logging.basicConfig(level=logging.INFO)

img_dims = (28, 28)  # X dimensions
y_dims = (10,)  # Y dimensions

def init() -> Tuple[Dict[str, Any], Dict[str, Array]]:
    """Initialize all model parameters in a single flat dictionary"""
    
    config = {
        "dataset_name": "mnist",
        "num_epochs": 5000,
        "hidden_dim": 256,
        "num_layers": 4,
        "time_embed_dim": 128,
        "num_train_images": 50000,  # Subset for demonstration
        "batch_size": 128,
        "learning_rate": 3e-4,
        "random_seed": 42,
        "vis_frequency": 100,  # Visualize every 100 epochs
        "optimizer": "adamw",
        "img_dims": img_dims,
        "timesteps": 1000,  # Number of diffusion timesteps
        "beta_start": 1e-4,
        "beta_end": 0.02,
    }
    
    key = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)
    
    hidden_dim = config["hidden_dim"]
    time_embed_dim = config["time_embed_dim"]
    num_layers = config["num_layers"]
    img_size = jnp.prod(jnp.array(img_dims))
    
    params = {}
    
    # Time embedding layers
    params['time_embed_1'] = jax.random.normal(next(key_gen).get(), (time_embed_dim, time_embed_dim)) * 0.02
    params['time_embed_1_bias'] = jnp.zeros(time_embed_dim)
    params['time_embed_2'] = jax.random.normal(next(key_gen).get(), (time_embed_dim, time_embed_dim)) * 0.02
    params['time_embed_2_bias'] = jnp.zeros(time_embed_dim)
    
    # Input projection
    params['input_proj'] = jax.random.normal(next(key_gen).get(), (img_size, hidden_dim)) * 0.02
    params['input_proj_bias'] = jnp.zeros(hidden_dim)
    
    # MLP layers
    for i in range(num_layers):
        # First layer takes hidden_dim + time_embed_dim as input
        input_dim = hidden_dim + time_embed_dim if i == 0 else hidden_dim
        params[f'mlp_{i}'] = jax.random.normal(next(key_gen).get(), (input_dim, hidden_dim)) * 0.02
        params[f'mlp_{i}_bias'] = jnp.zeros(hidden_dim)
    
    # Output projection
    params['output_proj'] = jax.random.normal(next(key_gen).get(), (hidden_dim, img_size)) * 0.02
    params['output_proj_bias'] = jnp.zeros(img_size)
    
    return config, params, key_gen

def get_beta_schedule(config: Dict[str, Any]) -> Array:
    """Get beta schedule for diffusion"""
    timesteps = config["timesteps"]
    beta_start = config["beta_start"]
    beta_end = config["beta_end"]
    
    # Linear schedule
    return jnp.linspace(beta_start, beta_end, timesteps)

def get_alpha_schedule(betas: Array) -> Tuple[Array, Array]:
    """Get alpha and alpha_cumprod schedules"""
    alphas = 1.0 - betas
    alpha_cumprod = jnp.cumprod(alphas)
    return alphas, alpha_cumprod

def time_embedding(t: Array, dim: int) -> Array:
    """Sinusoidal time embedding"""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t[..., None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return emb

def mlp_forward(params: Dict[str, Array], x: Array, t: Array) -> Array:
    """Forward pass through MLP denoising network"""
    # Time embedding - squeeze to remove batch dimension for single timestep
    t_emb = time_embedding(t, params['time_embed_1'].shape[0])
    if t_emb.ndim > 1:
        t_emb = jnp.squeeze(t_emb, axis=0)  # Remove batch dimension
    t_emb = jax.nn.silu(jnp.dot(t_emb, params['time_embed_1']) + params['time_embed_1_bias'])
    t_emb = jnp.dot(t_emb, params['time_embed_2']) + params['time_embed_2_bias']
    
    # Input projection
    x_flat = x.flatten()
    h = jax.nn.silu(jnp.dot(x_flat, params['input_proj']) + params['input_proj_bias'])
    
    # Concatenate time embedding
    h = jnp.concatenate([h, t_emb])
    
    # MLP layers
    for i in range(len([k for k in params.keys() if k.startswith('mlp_') and not k.endswith('_bias')])):
        h = jnp.dot(h, params[f'mlp_{i}']) + params[f'mlp_{i}_bias']
        h = jax.nn.silu(h)
    
    # Output projection
    output = jnp.dot(h, params['output_proj']) + params['output_proj_bias']
    return output.reshape(img_dims)

def q_sample(x_0: Array, t: Array, alpha_cumprod: Array, key: jax.random.PRNGKey) -> Tuple[Array, Array]:
    """Sample from q(x_t | x_0) - forward diffusion process"""
    sqrt_alpha_cumprod_t = jnp.sqrt(alpha_cumprod[t])
    sqrt_one_minus_alpha_cumprod_t = jnp.sqrt(1.0 - alpha_cumprod[t])
    
    noise = jax.random.normal(key, x_0.shape)
    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t, noise

def loss_fn_single(params: Dict[str, Array], x_0: Array, t: int, alpha_cumprod: Array, key: jax.random.PRNGKey) -> Array:
    """Compute loss for a single sample"""
    x_t, noise = q_sample(x_0, t, alpha_cumprod, key)
    predicted_noise = mlp_forward(params, x_t, jnp.array(t))  # Pass scalar timestep
    return jnp.mean((noise - predicted_noise) ** 2)

def loss_batch(params: Dict[str, Array], x_batch: Array, t_batch: Array, alpha_cumprod: Array, keys: Array) -> Array:
    """Compute loss for a batch"""
    loss_fn = jax.vmap(loss_fn_single, in_axes=(None, 0, 0, None, 0))
    return loss_fn(params, x_batch, t_batch, alpha_cumprod, keys)

@partial(jax.jit, static_argnums=(0,))
def train_step(optimizer: optax.GradientTransformation, params: Dict[str, Array], opt_state: Any, 
               x_batch: Array, t_batch: Array, alpha_cumprod: Array, keys: Array) -> Tuple[optax.Params, optax.OptState, Array]:
    
    def _loss_batch(p):
        return jnp.mean(loss_batch(p, x_batch, t_batch, alpha_cumprod, keys))
    
    loss_value, grads = jax.value_and_grad(_loss_batch)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value

def p_sample_step(params: Dict[str, Array], x_t: Array, t: int, betas: Array, alpha_cumprod: Array, key: jax.random.PRNGKey) -> Array:
    """Single denoising step"""
    if t == 0:
        noise = jnp.zeros_like(x_t)
    else:
        noise = jax.random.normal(key, x_t.shape)
    
    predicted_noise = mlp_forward(params, x_t, jnp.array(t))  # Pass scalar timestep
    
    alpha_t = 1.0 - betas[t]
    alpha_cumprod_t = alpha_cumprod[t]
    alpha_cumprod_t_prev = alpha_cumprod[t-1] if t > 0 else 1.0
    
    # Compute coefficients
    coeff1 = 1.0 / jnp.sqrt(alpha_t)
    coeff2 = betas[t] / jnp.sqrt(1.0 - alpha_cumprod_t)
    
    # Predicted x_0
    x_0_pred = coeff1 * (x_t - coeff2 * predicted_noise)
    
    # Compute posterior variance
    posterior_variance = betas[t] * (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t)
    
    return x_0_pred + jnp.sqrt(posterior_variance) * noise

def p_sample(params: Dict[str, Array], shape: Tuple[int, ...], timesteps: int, betas: Array, alpha_cumprod: Array, key: jax.random.PRNGKey) -> Array:
    """Full sampling process"""
    
    # Start from pure noise
    x_t = jax.random.normal(key, shape)
    
    # Reverse diffusion
    for t in range(timesteps - 1, -1, -1):
        x_t = p_sample_step(params, x_t, t, betas, alpha_cumprod, next(key_gen).get())
    
    return jnp.clip(x_t, 0.0, 1.0)

def test_sampling(params: Dict[str, Array], config: Dict[str, Any], betas: Array, alpha_cumprod: Array, key: jax.random.PRNGKey) -> Array:
    """Test sampling a few images"""
    images = []
    
    for i in range(6):  # Generate 6 random samples
        img = p_sample(params, img_dims, config["timesteps"], betas, alpha_cumprod, key)
        images.append(img)
    
    return jnp.stack(images)

if __name__ == "__main__":
    use_wandb = True
    
    # Initialize
    config, params_, key_gen = init()
    
    # Get diffusion schedules
    betas = get_beta_schedule(config)
    alphas, alpha_cumprod = get_alpha_schedule(betas)
    
    # Load dataset
    logging.info("Loading dataset...")
    data: ImageClassification = load_supervised_image(config["dataset_name"])
    data_X = data.X.reshape(data.n_samples, *img_dims) / 255.0
    data_test_X = data.X_test.reshape(data.n_test_samples, *img_dims) / 255.0
    
    optimizer = get_optimizer(config)
    
    # Initialize WandB if requested
    if use_wandb:
        logging.info("Initializing WandB...")
        wandb.init(entity="decode-transformer", project="diffusion-mnist", config=config)
    
    # Training loop
    logging.info("Training the diffusion model...")
    
    num_images = config["num_train_images"]
    opt_state_ = optimizer.init(params_)
    start_time = time.perf_counter()
    n_data_ = 0
    loss_history_ = []
    num_batches = num_images // config["batch_size"]
    
    for epoch in range(config["num_epochs"]):
        epoch_loss_ = 0.0
        key = next(key_gen).get()
        permutation = jax.random.permutation(key, num_images)
        shuffled_images = data_X[permutation]
        shuffled_labels = data.y[permutation]
        
        for batch in range(num_batches):
            _start = batch * config["batch_size"]
            _end = _start + config["batch_size"]
            
            # Get batch
            x_batch = shuffled_images[_start:_end]
            
            # Sample random timesteps
            t_batch = jax.random.randint(next(key_gen).get(), (config["batch_size"],), 0, config["timesteps"])
            
            # Generate noise keys for each sample
            noise_keys = jax.random.split(next(key_gen).get(), config["batch_size"])
            
            params_, opt_state_, _loss_value = train_step(optimizer, params_, opt_state_, 
                                                        x_batch, t_batch, alpha_cumprod, noise_keys)
            epoch_loss_ += _loss_value
        
        _avg_loss = epoch_loss_ / num_batches
        loss_history_.append(_avg_loss)
        n_data_ += num_images
        
        if use_wandb:
            wandb.log({
                "loss": _avg_loss,
                "epoch": epoch,
                "time": time.perf_counter() - start_time,
                "n_data": n_data_,
            })
        
        if epoch % 50 == 0:
            logging.info(f"Epoch {epoch}, Loss: {_avg_loss:.6f}")
        
        # Generate samples for visualization
        if use_wandb and (epoch % config["vis_frequency"] == 0 or epoch == config["num_epochs"] - 1):
            logging.info("Generating samples...")
            sample_key = next(key_gen).get()
            generated_images = test_sampling(params_, config, betas, alpha_cumprod, sample_key)
            
            # Log generated images
            wandb_images = []
            for i in range(6):
                img = generated_images[i]
                wandb_images.append(wandb.Image(jax.device_get(img), caption=f"Generated sample {i}"))
            
            wandb.log({"generated_samples": wandb_images})
            
            # Save a sample image locally
            plot_image(generated_images[0], title=f"Generated Sample Epoch {epoch}", 
                      save_path=f"./generated_sample_epoch_{epoch}.jpg")
    
    # Final sample generation
    logging.info("Generating final samples...")
    final_samples = test_sampling(params_, config, betas, alpha_cumprod, next(key_gen).get())
    
    # Save final samples
    for i in range(6):
        plot_image(final_samples[i], title=f"Final Generated Sample {i}", 
                  save_path=f"./final_sample_{i}.jpg")
    
    # Finish WandB run
    if use_wandb:
        wandb.finish()