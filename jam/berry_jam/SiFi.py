from functools import partial
import jax
import os, sys
import logging
from jax import Array
import jax.numpy as jnp
from typing import Dict, Any, Iterable, Tuple, List
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import optax




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.random_utils import infinite_safe_keys_from_key
from berries.my_datasets import ImageClassification, load_supervised_image 
from pos_net import create_coordinate_indices, make_sinusoidal_pos_embed


logging.basicConfig(level=logging.INFO)

img_dims = (28, 28)  # X dimensions
pos_embed_dim = 16
K = 2 
lr = 0.001

def init() -> Tuple[Dict[str, Any], Dict[str, Array], optax.GradientTransformation]:
    """Initialize all model parameters in a single flat dictionary"""

    config = {
        "dataset_name": "fashion_mnist",
        "num_epochs": 10000,
        "pos_embed_dim": pos_embed_dim,  # Positional embedding dimension
        "item_embed_dim": 8,  # Item embedding dimension
        "Fi_Q_count": 10,
        "random_seed": 42,
        "qk_dim": 8,
        "vis_frequency": 200,  # Visualize every 200 epochs
        "lr": lr,
        "img_dims": img_dims,  # Image dimensions
        "K": K,
    }
    key = jax.random.PRNGKey(config["random_seed"])
    item_embed_dim = config["item_embed_dim"]
    qk_dim = config["qk_dim"]
    key_gen = infinite_safe_keys_from_key(key)

    params = {
        'pos_embedding': jax.random.normal(next(key_gen).get(), (*img_dims, pos_embed_dim)),
        'input_proj': jax.random.normal(next(key_gen).get(), (pos_embed_dim + 1, item_embed_dim)) * 0.01,
        'SiFi_q': jax.random.normal(next(key_gen).get(), (qk_dim, config['Fi_Q_count'])) * 0.01,
        'SiFi_Wk': jax.random.normal(next(key_gen).get(), (item_embed_dim, qk_dim)) * 0.01,
        'SiFi_Wv': jax.random.normal(next(key_gen).get(), (item_embed_dim, item_embed_dim)) * 0.01,
        'SiFi_b': jax.random.normal(next(key_gen).get(), (item_embed_dim,)) * 0.01,
        'SiFi_Wo': jax.random.normal(next(key_gen).get(), (item_embed_dim, item_embed_dim)) * 0.01,
        'SiFi_W': jax.random.normal(next(key_gen).get(), (item_embed_dim * K * config['Fi_Q_count'],  config['Fi_Q_count'] * item_embed_dim )) * 0.01,
    }
    
    # Create SGD optimizer
    optimizer = optax.sgd(config["lr"])
    
    return config, params, key_gen, optimizer

def get_positional_embedding(params: Dict[str, Array], coords: Iterable[int]) -> Array:
    """Get embeddings for coordinates"""
    # Extract coordinates
    intrinsic_pos_embed = params['pos_embedding'][*coords]
    sino_pos_embed = make_sinusoidal_pos_embed(img_dims, d_model=pos_embed_dim)[*coords]
    return 0.01 * (intrinsic_pos_embed + sino_pos_embed)


def get_positional_embeddings(params: Dict[str, Array], coords: Array) -> Array:
    """Get embeddings for coordinates"""
    return jax.vmap(get_positional_embedding, in_axes=(None, 0))(params, coords)

def get_item_embedding_1(params: Dict[str, Array], x: Array, coords: Array) -> Array:
    """Get embeddings for items"""
    pos_embed = get_positional_embedding(params, coords)
    x_1 = jnp.atleast_1d(x[*coords])
    x_embed = jnp.concatenate([pos_embed, x_1], axis=0)
    x_embed = jnp.dot(x_embed, params['input_proj'])
    return x_embed

def get_item_embeddings(params: Dict[str, Array], X: Array) -> Array:
    """Get embeddings for items"""
    coords = create_coordinate_indices(*img_dims)
    return jax.vmap(get_item_embedding_1, in_axes=(None, None, 0))(params, X, coords)

def get_focused_embeddings_indices(params: Dict[str, Array], X: Array, qs: Array) -> Array:
    """Get embeddings for items"""
    item_embeddings = get_item_embeddings(params, X)
    # get topk items that are closest to each q in qs
    def get_topk_for_single_q(q):
        return jnp.argsort(jnp.linalg.norm(item_embeddings - q, axis=1))[:K]
    
    topk_indices = jax.vmap(get_topk_for_single_q)(qs)
    return topk_indices

def get_focused_embeddings(params: Dict[str, Array], X: Array, q: Array) -> Array:
    """Get embeddings for items"""
    item_embeddings = get_item_embeddings(params, X)
    topk_indices = get_focused_embeddings_indices(params, X, q)
    return item_embeddings[topk_indices]

def predict_query_attn(params: Dict[str, Array], focused_item_embeddings: Array) -> Array:
    """Predict the query from the item embeddings (Fi_Q_count, K, item_embed_dim)"""
    # combine the first two dimensions
    focused_item_embeddings = jax.lax.collapse(focused_item_embeddings, 0, 2)
    q = params['SiFi_q']   # (qk_dim, Fi_Q_count)
    ks = focused_item_embeddings @ params['SiFi_Wk'] # (Fi_Q_count * K, qk_dim)
    vs = focused_item_embeddings @ params['SiFi_Wv'] # (Fi_Q_count * K, item_embed_dim)
    attn = ks @ q # (Fi_Q_count * K, Fi_Q_count)
    attn = jax.nn.softmax(attn.T, axis=-1) # (Fi_Q_count, Fi_Q_count * K)
    attn_out = attn @ vs  # (Fi_Q_count, item_embed_dim)
    attn_out = jax.nn.swish(attn_out)
    out = attn_out @ params['SiFi_Wo'] # (Fi_Q_count, item_embed_dim)
    return out # (Fi_Q_count, item_embed_dim)

def predict_query_linear(params: Dict[str, Array], focused_item_embeddings: Array) -> Array:
    """Predict the query from the item embeddings (Fi_Q_count * K, item_embed_dim)"""
    out = focused_item_embeddings.flatten() @ params['SiFi_W']
    return out.reshape(config['Fi_Q_count'], -1) # (Fi_Q_count, item_embed_dim)

predict_query = predict_query_attn

def predict_loss(params: Dict[str, Array], X: Array, q: Array) -> Array:
    """Predict the loss from the item embeddings (K, item_embed_dim)"""
    focused_embeddings = get_focused_embeddings(params, X, q)
    predicted_q = predict_query(params, focused_embeddings)
    loss = jnp.linalg.norm(predicted_q - q)
    return loss

def mean_loss_minibatch(params: Dict[str, Array], X: Array, qs: Array) -> Array:
    """Mean loss over a minibatch"""
    loss_minibatch = jax.vmap(predict_loss, in_axes=(None, None, 0), out_axes=0)
    return jnp.mean(loss_minibatch(params, X, qs))

@partial(jax.jit, static_argnums=(0,))
def update_params(optimizer: optax.GradientTransformation, params: Dict[str, Array], opt_state: Any, X: Array, qs: Array) -> Tuple[Dict[str, Array], Any]:
    """Update only the SiFi parameters using optax SGD"""
    grads = jax.grad(mean_loss_minibatch)(params, X, qs)
    
    # Filter gradients to only update SiFi parameters
    filtered_grads = {}
    for key, grad in grads.items():
        if key.startswith('SiFi'):
            filtered_grads[key] = grad
        else:
            filtered_grads[key] = jnp.zeros_like(grad)
    # Apply optimizer updates
    updates, new_opt_state = optimizer.update(filtered_grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

def update_params_1(optimizer: optax.GradientTransformation, params: Dict[str, Array], opt_state: Any, X: Array, q: Array) -> Tuple[Dict[str, Array], Any]:
    """Update only the SiFi parameters using optax SGD"""
    qs = jnp.array([q])
    return update_params(optimizer, params, opt_state, X, qs)


def visualize_item_embeddings(params: Dict[str, Array], original_image: Array, save_path: str = "item_embeddings.png"):
    """Calculate and visualize item embeddings and original image in a 3x3 grid"""
    # Calculate item embeddings
    item_embeddings = get_item_embeddings(params, original_image)
    print(f"Item embeddings shape: {item_embeddings.shape}")
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Plot embeddings 0-7 around the edges
    for i in range(8):
        j = i + 1 if i >= 4 else i
        row = j // 3
        col = j % 3
        axes[row, col].imshow(item_embeddings[:, i].reshape(img_dims))
        axes[row, col].set_title(f'Embedding {i}')
        axes[row, col].axis('off')
    
    # Plot original image in center
    axes[1, 1].imshow(original_image)
    axes[1, 1].set_title('Original Image')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    
    return item_embeddings

def visualize_focus_animation(params: Dict[str, Array], original_sample: Array, q_sequence: List[Array], save_path: str = "focused_animation.mp4"):
    """Create an animation showing how focus changes across a sequence of queries"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Initialize plots
    im1 = ax1.imshow(original_sample)
    ax1.set_title('Original Sample')
    ax1.axis('off')
    
    im2 = ax2.imshow(original_sample)
    ax2.set_title('Focused Pixels')
    ax2.axis('off')
    
    # Create colormap for highlighting
    c_white = matplotlib.colors.colorConverter.to_rgba('white', alpha=0)
    c_black = matplotlib.colors.colorConverter.to_rgba('white', alpha=1)
    cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap', [c_white, c_black], 512)
    
    # Initialize overlay
    focused_sample = jnp.zeros_like(original_sample).flatten()
    focused_sample = focused_sample.at[get_focused_embeddings_indices(params, original_sample, q_sequence[0])].set(1)
    focused_sample = focused_sample.reshape(img_dims)
    overlay = ax2.imshow(1 - focused_sample, cmap=cmap_rb, alpha=0.95)
    
    def animate(frame):
        q = q_sequence[frame]
        focused_indices = get_focused_embeddings_indices(params, original_sample, q)
        
        # Create focused sample mask
        focused_sample = jnp.zeros_like(original_sample).flatten()
        focused_sample = focused_sample.at[focused_indices].set(1)
        focused_sample = focused_sample.reshape(img_dims)
        
        # Update the overlay
        overlay.set_array(1 - focused_sample)
        
        ax2.set_title(f'Focused Pixels (Frame {frame + 1}/{len(q_sequence)})')
        
        return [im1, im2, overlay]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(q_sequence), 
                        interval=500, blit=True, repeat=True)
    
    # Save animation as MP4
    anim.save(save_path, writer='ffmpeg', fps=2, dpi=100)
    plt.close()
    
    return anim

def visualize_focus(params: Dict[str, Array], original_sample: Array, q: Array, save_path: str = "focused_sample.png"):
    """Visualize the focused pixels based on query embedding (static version)"""
    focused_indices = get_focused_embeddings_indices(params, original_sample, q)
    print(focused_indices)
    
    # Create focused sample in flattened state first, then reshape
    focused_sample = jnp.zeros_like(original_sample).flatten()
    focused_sample = focused_sample.at[focused_indices].set(1)
    focused_sample = focused_sample.reshape(img_dims)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original_sample)
    ax1.set_title('Original Sample')
    ax1.axis('off')
    
    # Show original image with focused pixels highlighted
    ax2.imshow(original_sample)
    c_white = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0)
    c_black= matplotlib.colors.colorConverter.to_rgba('white',alpha = 1)
    cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_black],512)

    ax2.imshow(1 - focused_sample, cmap=cmap_rb, alpha=0.95)
    ax2.set_title('With the focused pixels revealing')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    
    return focused_indices

if __name__ == "__main__":
    use_wandb = True

    # Define configuration dictionary with all parameters
    config, params_, key_gen, optimizer = init()

    # Load dataset
    logging.info("Loading dataset...")
    data: ImageClassification = load_supervised_image(config["dataset_name"], n_tr=10, n_tst=1)
    # Flatten images and normalize pixel values to [0, 1]
    data_X = data.X.reshape(data.n_samples, *img_dims) / 255.0
    data_test_X = data.X_test.reshape(data.n_test_samples, *img_dims) / 255.0

    the_sample = data_X[0]

    # visualize_item_embeddings(params_, the_sample)
    item_embeddings = get_item_embeddings(params_, the_sample)
    print(item_embeddings.shape)
    
    # Use a random query instead of one from the same image
    fi_ = jax.random.normal(next(key_gen).get(), (config['Fi_Q_count'], config['item_embed_dim'])) * 0.01  # Random query with small magnitude
    
    # Collect query sequence for animation
    query_sequence = [fi_.copy()]
    
    # focused_indices = visualize_focus(params_, the_sample, q)
    visualize_focus(params_, the_sample, fi_, f"focused_sample_{0}.png")
    
    # Initialize optimizer state
    opt_state = optimizer.init(params_)
    init_fis = []
    init_sis = []
    for i in range(100):
        fi_random = jax.random.normal(next(key_gen).get(), fi_.shape) * 0.01
        init_fis.append(fi_random)
    init_fis = jnp.array(init_fis)
    for j in range(100):
        loss = mean_loss_minibatch(params_, the_sample, init_fis)
        print(f"Loss: {loss}")
        params_, opt_state = update_params(optimizer, params_, opt_state, the_sample, init_fis)
        loss_after = mean_loss_minibatch(params_, the_sample, init_fis)
        print(f"Loss after update: {loss_after}")
    
    for i in range(1, 60):
        # r_ = data_X[i % 3]
        r_ = the_sample
        for j in range(1):
            loss = predict_loss(params_, r_, fi_)
            # print(f"Loss: {loss}")
            params_, opt_state = update_params_1(optimizer, params_, opt_state, r_, fi_)
            loss_after = predict_loss(params_, r_, fi_)
            # print(f"Loss after update: {loss_after}")

        # Update the query to be the predicted query from the previous iteration
        # fi_ = fi_ + jax.random.normal(next(key_gen).get(), fi_.shape) * 0.01
        print(f"real fi_: {fi_[0, :]}")
        si_ = get_focused_embeddings(params_, r_, fi_)
        print(f"si_: {si_[0, :]}")
        fi_ = predict_query(params_, si_)
        print(f"fi_: {fi_[0, :]}")
        query_sequence.append(fi_.copy())
        # visualize_focus(params_, r_, fi_, f"focused_sample_{i}.png")
    
    # Create animation from the collected query sequence
    print("Creating animation...")
    visualize_focus_animation(params_, the_sample, query_sequence, "focused_animation.mp4")



    # Initialize WandB if requested
    # if use_wandb:
    #     logging.info("Initializing WandB...")
    #     wandb.init(entity="decode-transformer", project="image-reconstruction-embeddings", config=config)

    # Train the model 
    # logging.info("Training the model...")
