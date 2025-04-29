from functools import partial
import time
import jax
from jax import Array, jit
import jax.numpy as jnp
import optax
from typing import Dict, List, Any, Tuple

from pos_net import create_coordinate_indices

from wandb_vis.embed_utils import log_visualizations
from wandb_vis.shared_utils import log_epoch_metrics
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.random_utils import infinite_safe_keys, infinite_safe_keys_from_key



def init_model_params(key: Array, num_images: int, img_dim: int, hidden_dim: int = 32, 
                     embed_dim: int = 16, qk_dim: int= 16) -> Dict[str, Array]:
    """Initialize all model parameters in a single flat dictionary"""
    key_gen = infinite_safe_keys_from_key(key)
    
    # Init parameters in a flat dictionary
    params = {}
    
    # Height network params
    params['h_W1'] = jax.random.normal(next(key_gen).get(), (1, hidden_dim)) * jnp.sqrt(2.0) * 5 
    params['h_b1'] = jnp.zeros((hidden_dim,))
    params['h_W2'] = jax.random.normal(next(key_gen).get(), (hidden_dim, embed_dim)) * jnp.sqrt(2.0 / hidden_dim)
    params['h_b2'] = jnp.zeros((embed_dim,))
    
    # Width network params
    params['w_W1'] = jax.random.normal(next(key_gen).get(), (1, hidden_dim)) * jnp.sqrt(2.0) * 5 
    params['w_b1'] = jnp.zeros((hidden_dim,))
    params['w_W2'] = jax.random.normal(next(key_gen).get(), (hidden_dim, embed_dim)) * jnp.sqrt(2.0 / hidden_dim)
    params['w_b2'] = jnp.zeros((embed_dim,))
    
    params['pos_embedding'] = jax.random.normal(next(key_gen).get(), (28 * 28, embed_dim * 2)) * 0.01
    # params['pos_embedding'] = jnp.ones((28 * 28, embed_dim * 2)) * 0.01
    # Image embeddings
    # params['img_map'] = jax.random.normal(next(key_gen).get(), (img_dim, embed_dim * 2)) * jnp.sqrt(2.0 / img_dim)
    params['E_w1'] = jax.random.normal(next(key_gen).get(), (img_dim, hidden_dim)) * jnp.sqrt(2.0 / img_dim)
    params['E_b1'] = jnp.zeros((hidden_dim,))
    params['E_w2'] = jax.random.normal(next(key_gen).get(), (hidden_dim, embed_dim * 2)) * jnp.sqrt(2.0 / hidden_dim)
    params['E_b2'] = jnp.zeros((embed_dim * 2,))

    params['img_embed'] = jax.random.normal(next(key_gen).get(), (num_images, embed_dim * 2)) * 0.01
    # params['img_embed'] = jnp.ones((num_images, embed_dim * 2)) * 0.01

    out_hidden_dim = 512
    params['w_out'] = jax.random.normal(next(key_gen).get(), (embed_dim * 2,)) * 0.01
    params['w_out_u'] = jax.random.normal(next(key_gen).get(), (embed_dim * 2, out_hidden_dim)) * 0.01
    params['w_out_v'] = jax.random.normal(next(key_gen).get(), (out_hidden_dim, embed_dim * 2)) * 0.01

    params['b_out'] = jnp.zeros((1,))
    # params['w_out1'] = jax.random.normal(key4, (embed_dim * 2, out_hidden_dim)) * jnp.sqrt(2.0 / (embed_dim * 2))
    # params['w_out2'] = jax.random.normal(key5, (out_hidden_dim, 1)) * jnp.sqrt(2.0 / out_hidden_dim)

    # params['w_out_c'] = jax.random.normal(key5, (embed_dim * 4,)) * jnp.sqrt(2.0 / (embed_dim * 4))

    params['w_out_c1'] = jax.random.normal(next(key_gen).get(), (embed_dim * 4, out_hidden_dim)) * jnp.sqrt(2.0 / (embed_dim * 4))
    params['w_out_c2'] = jax.random.normal(next(key_gen).get(), (out_hidden_dim, 1)) * jnp.sqrt(2.0 / out_hidden_dim)

    params['q'] = jax.random.normal(next(key_gen).get(), (2, qk_dim)) * 0.01
    params['k'] = jax.random.normal(next(key_gen).get(), (2, qk_dim)) * 0.01
    params['v'] = jax.random.normal(next(key_gen).get(), (2, qk_dim)) * 0.01
    params['q_out'] = jax.random.normal(next(key_gen).get(), (qk_dim,)) * 0.01
    
    return params


def pos_network_forward(params: Tuple[Array, ...], coord: Array) -> Array:
    """Forward pass through positional embedding network for one dimension"""
    W1, b1, W2, b2 = params
    # First layer with sine activation
    h = jnp.dot(coord, W1) + b1
    h = jnp.sin(h)
    
    # Second layer
    y = jnp.dot(h, W2) + b2
    
    return y


@jit
def get_positional_embeddings(h_W1: Array, h_b1: Array, h_W2: Array, h_b2: Array,
                           w_W1: Array, w_b1: Array, w_W2: Array, w_b2: Array, 
                           coords: Array, pos_embedding: Array, **_) -> Array:
    """Get embeddings for coordinates using positional networks"""
    # Extract coordinates
    y_coords, x_coords  = coords[:, 0:1], coords[:, 1:2]
    
    y_embeddings = pos_network_forward((h_W1, h_b1, h_W2, h_b2), y_coords)
    x_embeddings = pos_network_forward((w_W1, w_b1, w_W2, w_b2), x_coords)
    
    # return jnp.concatenate([y_embeddings, x_embeddings], axis=1) + pos_embedding
    return pos_embedding

def get_image_embedding(params: Dict[str, Array], target: Array, img_embed: Array) -> Array:
    """Get the image embedding for a given target"""
    # Compute the image embedding
    # img_emb = target @ params['img_map'] 
    # img_embedding = jax.nn.swish(target @ params['E_w1'] + params['E_b1']) @ params['E_w2'] + params['E_b2']
    # return img_embedding
    return img_embed

def pix_val_forward(img_embed: Array, pos_embedding: Array, p: Dict[str, Array]) -> Array:
    """Compute pixel values using the image embedding and positional embeddings"""
    # Compute pixel values
    # pixel_values = jax.nn.swish((pos_embedding + img_embed) @ p['w_out1']) @ p['w_out2'] + p['b_out']
    # pixel_values = jnp.dot(pos_embedding, img_embed) 
    # spilit both in half, dot and add
    # pos_embedding_1, pos_embedding_2 = jnp.split(pos_embedding, 2)
    # img_embed_1, img_embed_2 = jnp.split(img_embed, 2)
    # pixel_values_1 = jnp.dot(pos_embedding_1, img_embed_1)
    # pixel_values_2 = jnp.dot(pos_embedding_2, img_embed_2)
    # return jax.nn.sigmoid(jnp.mean(pixel_values_1)) + jax.nn.sigmoid(jnp.mean(pixel_values_2))
    pixel_values = jnp.dot(pos_embedding, img_embed) #+ p['b_out']
    # pixel_values = jnp.concatenate([img_embed, pos_embedding], axis=0) @ p['w_out_c'] + p['b_out']
    # pixel_values = jax.nn.swish(jnp.concat([img_embed, pos_embedding], axis=0) @ p['w_out_c1']) @ p['w_out_c2'] + p['b_out']
    # pixel_values = jax.nn.sigmoid(pixel_values)
    # k = v = jnp.stack([img_embed, pos_embedding], axis=0).T.reshape(-1, 1, 2)
    # b = jnp.stack([img_embed, pos_embedding], axis=0).T.reshape(-1, 1, 2)
    # k = b @ p['k']
    # v = b @ p['v']
    # att = jax.nn.dot_product_attention(p['q'].reshape(2, 1, -1), k, v) 
    # pixel_values = att.reshape(2, -1) @ p['q_out']
    # pixel_values = jax.nn.softmax(pixel_values)[1]
    # return pixel_values
    return jax.nn.sigmoid(jnp.mean(pixel_values) + p['b_out'])


def predict_image(params: Dict[str, Array], img_idx: int, coords: Array) -> Array:
    """Predict pixel values for a single image"""
    pos_embeddings = get_positional_embeddings(**params, coords=coords)
    # Get the embedding for this image
    img_embedding = params['img_embed'][img_idx]
    # return pix_val_forward(img_embedding, pos_embeddings, params)
    return jax.vmap(pix_val_forward, in_axes=(None, 0, None))(img_embedding, pos_embeddings, params)


def debug_jax(
    key: Array, num_images: int, img_dim, hidden_dim: int = 32, embed_dim: int = 16, qk_dim: int = 16):
    params = init_model_params(key, num_images, img_dim, hidden_dim, embed_dim, qk_dim)
    coords = create_coordinate_indices(28, 28)
    pixel_values = predict_image(params, 0, coords)
    return pixel_values

@partial(jax.jit, static_argnums=(0, ))
def train_step(optimizer: optax.GradientTransformation, coords: Array, params: Dict[str, Array], opt_state: Any, indices: Array, batch_images: Array) -> Tuple[optax.Params, optax.OptState, Array]:
    # Use the provided img_indices, which now refer to the fixed-order embeddings.
    def loss_batch(p):
        batch_img_emb = p['img_embed'][indices] 

        def loss_pixel(img_emb, pos_emb, target_pixel):
            predicted =  pix_val_forward(img_emb, pos_emb, p)
            return (predicted - target_pixel) ** 2

        # Compute loss for each image in the batch
        def loss_1(img_emb, target):
            img_emb = get_image_embedding(p, target, img_emb)
            pos_embs = get_positional_embeddings(**p, coords=coords)
            return jax.vmap(loss_pixel, in_axes=(None, 0, 0))(img_emb, pos_embs, target)
        return jnp.mean(jax.vmap(loss_1, in_axes=(0, 0))(batch_img_emb, batch_images))
    
    # loss_fn = jax.vmap(loss_1, in_axes=([0, None], [None, 0], [0, 1], None))

    loss_value, grads = jax.value_and_grad(loss_batch)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value

def train_model(key: Array, coords: Array, flattened_images: Array, 
                           num_images: int, hidden_dim: int = 32, embed_dim: int = 16, 
                           qk_dim: int = 16, num_epochs: int = 200, batch_size: int = 10, 
                           optimizer: optax.GradientTransformation = optax.adam(1e-3),
                           use_wandb: bool = False, vis_frequency: int = 50,
                           height: int = 28, width: int = 28,
                           data: Any = None) -> Tuple[Dict[str, Array], List[float]]:
    
    # Initialize parameters in a fixed order.
    params = init_model_params(key, num_images, height * width, hidden_dim, embed_dim, qk_dim=qk_dim)
    
    # Initialize optimizer state—this will remain in sync with the fixed-order params.
    opt_state = optimizer.init(params)
    
    # Create JIT-compiled training step function
    start = time.perf_counter()
    n_data = 0
    
    loss_history = []
    num_batches = num_images // batch_size
    
    all_indices = jnp.arange(num_images)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        _, key = jax.random.split(key)
        permutation = jax.random.permutation(key, num_images)
        shuffled_images = flattened_images[permutation]
        shuffled_indices = all_indices[permutation]


        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            indices = shuffled_indices[start_idx:end_idx]
            
            params, opt_state, loss = train_step(optimizer, coords, params, opt_state,  
                                                 indices, shuffled_images[start_idx:end_idx])
            epoch_loss += loss #.block_until_ready()
        
        # Average loss for the epoch
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        n_data += num_images
        # Log to WandB
        if use_wandb:
            log_epoch_metrics(epoch, avg_loss, n_data, int(time.perf_counter() - start))
            def new_predict_image(_, img_idx, __):
                pos_embeddings = get_positional_embeddings(**params, coords=coords)
                target = flattened_images[img_idx]
                img_emb = get_image_embedding(params, target, params['img_embed'][img_idx])
                pixel_values = jax.vmap(pix_val_forward, in_axes=(None, 0, None))(img_emb, pos_embeddings, params)
                return pixel_values

            
            if epoch % vis_frequency == 0 or epoch == num_epochs - 1:
                log_visualizations(
                    epoch=epoch,
                    params=params, 
                    coords=coords, 
                    flattened_images=flattened_images, 
                    height=height, 
                    width=width, 
                    pos_network_forward=pos_network_forward,
                    predict_image=jit(new_predict_image),
                    img_embed=jit(get_image_embedding)(params, flattened_images, params['img_embed']), 
                    data=data, 
                    num_train_images=num_images
                )
                
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    return params, loss_history