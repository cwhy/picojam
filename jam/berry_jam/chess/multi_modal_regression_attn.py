import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import wandb
import os, sys
import numpy as np

# Add berries to path (from your existing code structure)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from berries.my_datasets import ImageClassification, load_supervised_image

# CNN configuration with attention mechanism
cnn_config = {
    'conv1_features': 32,
    'conv2_features': 64, 
    'conv3_features': 128,
    'n_modes': 16,        # Number of attention modes/keys
    'hidden_dim': 1024,    # Hidden dimension for query/key projection
    'output_features': 1
}

# Initialize wandb
wandb.init(
    entity="decode-transformer",
    project="multi-mode-regression-attention",
    config={
        "learning_rate": 0.001,
        "cnn_config": cnn_config,
        "n_steps": 20000,
        "batch_size": 512,
        "n_samples": 10000,
        "log_freq": 50,
        "img_size": (28, 28),
        "weight_decay": 0.0001
    }
)

def create_multimodal_targets(mnist_labels, mnist_images):
    """Create dramatically multimodal continuous targets with 3 well-separated modes"""
    # Flatten images for pixel intensity calculations
    flat_images = mnist_images.reshape(len(mnist_images), -1)
    
    # Create multimodal targets with 3 VERY separated modes
    targets = jnp.zeros(len(mnist_labels))
    
    for digit in range(10):
        mask = mnist_labels == digit
        if jnp.sum(mask) > 0:
            # Group digits into 3 well-separated modes - EXACTLY as in original
            if digit in [1, 2, 3, 5]:  # Mode around -30
                base_target = -30.0
            elif digit in [6, 9]:  # Mode around 0
                base_target = 0.0
            else:  # digits 0, 4, 7, 8 - Mode around +30
                base_target = 30.0
            
            # Add small image-dependent variation within each mode
            pixel_intensity = jnp.mean(flat_images[mask], axis=1)
            variation = (pixel_intensity - 0.5) * 0.1  # Small variation to keep modes distinct
            
            targets = targets.at[mask].set(base_target + variation)
    
    return targets

def init_cnn_with_attention(config, key):
    """Initialize CNN with attention mechanism parameters"""
    keys = jax.random.split(key, 10)  # Need more keys for attention components
    
    # Conv layer 1: 1 -> 32 channels, 5x5 kernel
    conv1_w = jax.random.normal(keys[0], (5, 5, 1, config['conv1_features'])) * jnp.sqrt(2.0 / (5*5*1))
    conv1_b = jnp.zeros(config['conv1_features'])
    
    # Conv layer 2: 32 -> 64 channels, 5x5 kernel  
    conv2_w = jax.random.normal(keys[1], (5, 5, config['conv1_features'], config['conv2_features'])) * jnp.sqrt(2.0 / (5*5*config['conv1_features']))
    conv2_b = jnp.zeros(config['conv2_features'])
    
    # Conv layer 3: 64 -> 128 channels, 3x3 kernel
    conv3_w = jax.random.normal(keys[2], (3, 3, config['conv2_features'], config['conv3_features'])) * jnp.sqrt(2.0 / (3*3*config['conv2_features']))
    conv3_b = jnp.zeros(config['conv3_features'])
    
    # Calculate flattened size after conv layers
    flatten_size = 3 * 3 * config['conv3_features']  # 1152
    
    # Attention mechanism components
    # Query projection: features -> query vector
    query_w = jax.random.normal(keys[4], (flatten_size, config['hidden_dim'])) * jnp.sqrt(2.0 / flatten_size)
    
    # Key matrix: learnable keys for different modes
    keys_matrix = jax.random.normal(keys[5], (config['n_modes'], config['hidden_dim'])) * jnp.sqrt(2.0 / config['hidden_dim'])
    
    # Value matrix: learnable values (the actual regression values for each mode)
    # values_matrix = jax.random.normal(keys[6], (config['n_modes'], 1)) * 10 # Initialize with larger values for regression
    values_matrix = jnp.linspace(-35.0, 35.0, config['n_modes']).reshape(-1, 1)
    
    # Optional: Add a small MLP after attention for fine-tuning
    attention_mlp_w = jax.random.normal(keys[7], (1, config['output_features'])) * jnp.sqrt(2.0)
    attention_mlp_b = jnp.zeros(config['output_features']) 
    
    return {
        'conv1': (conv1_w, conv1_b),
        'conv2': (conv2_w, conv2_b), 
        'conv3': (conv3_w, conv3_b),
        'query_w': query_w,
        'keys': keys_matrix,
        'values': values_matrix,
        'attention_mlp': (attention_mlp_w, attention_mlp_b)
    }

def attention_mechanism(query, keys, values):
    """
    Compute attention weights and output
    query: (hidden_dim,) - query vector from input features
    keys: (n_modes, hidden_dim) - learnable key vectors
    values: (n_modes, 1) - learnable value scalars for each mode
    """
    # Compute attention scores: query · keys^T
    scores = jnp.dot(keys, query)  # (n_modes,)
    
    # Apply softmax to get attention weights
    attention_weights = jax.nn.softmax(scores)  # (n_modes,)

    # Normalize scores
    # scores = scores / jnp.sqrt(cnn_config['hidden_dim'])
    
    # Weighted sum of values
    output = jnp.dot(attention_weights, values.squeeze())  # scalar
    
    return output, attention_weights

def cnn_forward_with_attention(params, x):
    """Forward pass through CNN with attention mechanism"""
    # Add batch and channel dimensions: (28, 28) -> (1, 28, 28, 1)
    x = x.reshape(1, 28, 28, 1)
    
    # Conv layer 1 + ReLU + MaxPool
    conv1_w, conv1_b = params['conv1']
    x = jax.lax.conv_general_dilated(
        x, conv1_w, 
        window_strides=[1, 1], 
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = jax.nn.relu(x + conv1_b.reshape(1, 1, 1, -1))
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, 
        window_dimensions=(1, 2, 2, 1), 
        window_strides=(1, 2, 2, 1), 
        padding='VALID'
    )
    
    # Conv layer 2 + ReLU + MaxPool  
    conv2_w, conv2_b = params['conv2']
    x = jax.lax.conv_general_dilated(
        x, conv2_w, 
        window_strides=[1, 1], 
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = jax.nn.relu(x + conv2_b.reshape(1, 1, 1, -1))
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, 
        window_dimensions=(1, 2, 2, 1), 
        window_strides=(1, 2, 2, 1), 
        padding='VALID'
    )
    
    # Conv layer 3 + ReLU + MaxPool
    conv3_w, conv3_b = params['conv3']
    x = jax.lax.conv_general_dilated(
        x, conv3_w, 
        window_strides=[1, 1], 
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    x = jax.nn.relu(x + conv3_b.reshape(1, 1, 1, -1))
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, 
        window_dimensions=(1, 2, 2, 1), 
        window_strides=(1, 2, 2, 1), 
        padding='VALID'
    )
    
    # Flatten features
    x = x.reshape(-1)  # (1152,)
    
    # Project to feature dimension
    query = jnp.tanh(x @ params['query_w'])   # (hidden_dim,)
    
    # Apply attention mechanism
    keys = params['keys']  # (n_modes, hidden_dim)
    values = params['values']  # (n_modes, 1)
    
    attention_output, attention_weights = attention_mechanism(query, keys, values)
    
    # Optional: Pass through small MLP for final adjustment
    attention_mlp_w, attention_mlp_b = params['attention_mlp']
    final_output = attention_output * attention_mlp_w.squeeze() + attention_mlp_b.squeeze()
    # final_output = attention_output
    
    return final_output

# Loss function
def loss_fn(params, x_batch, y_batch):
    """Compute MSE loss for batch"""
    pred = vmap(lambda x: cnn_forward_with_attention(params, x))(x_batch)
    return jnp.sqrt(jnp.mean((pred - y_batch) ** 2))

# Load MNIST data
print("Loading MNIST dataset...")
data: ImageClassification = load_supervised_image("mnist")
mnist_images = data.X.reshape(data.n_samples, 28, 28) / 255.0
mnist_labels = data.y

# Split data into train/validation sets
n_samples = wandb.config.n_samples
total_indices = jnp.arange(min(n_samples, len(mnist_images)))
n_train = int(0.8 * len(total_indices))

train_indices = total_indices[:n_train]
val_indices = total_indices[n_train:]

# Training data
x_train = mnist_images[train_indices]
train_labels = mnist_labels[train_indices]
y_train = create_multimodal_targets(train_labels, data.X[train_indices].reshape(len(train_indices), 28, 28))

# Validation data  
x_val = mnist_images[val_indices]
val_labels = mnist_labels[val_indices]
y_val = create_multimodal_targets(val_labels, data.X[val_indices].reshape(len(val_indices), 28, 28))

print(f"Train data shape: {x_train.shape}, Val data shape: {x_val.shape}")

# Use training data for main training
x_data = x_train
y_data = y_train
labels = train_labels

def evaluate_and_log(params, step, x_val, y_val, val_labels, suffix=""):
    """Evaluate model on validation set and log scatter plot + attention analysis"""
    # Get predictions on validation set
    y_pred_val = vmap(lambda x: cnn_forward_with_attention(params, x))(x_val)
    val_mse = jnp.mean((y_pred_val - y_val) ** 2)
    
    # Create scatter plot data grouped by digit
    scatter_data = []
    for i, (yt, yp, label) in enumerate(zip(y_val, y_pred_val, val_labels)):
        scatter_data.append([float(yt), float(yp), int(label)])
    
    table = wandb.Table(data=scatter_data, columns=["y_true", "y_pred", "digit"])
    
    wandb.log({
        f"val_predictions_scatter{suffix}": wandb.plot.scatter(
            table, "y_true", "y_pred", 
            title=f"Validation: Predictions vs True Values (Step {step})"
        ),
        f"val_mse{suffix}": float(val_mse)
    })
    
    # Log learned attention values/modes
    learned_values = params['values'].squeeze()
    wandb.log({
        f"learned_mode_values{suffix}": wandb.Histogram(np.array(learned_values)),
        f"mode_value_range{suffix}": float(jnp.max(learned_values) - jnp.min(learned_values))
    })
    
    return val_mse

print(f"Train data shape: {x_data.shape}, Val data shape: {x_val.shape}")
print(f"Target statistics - Mean: {jnp.mean(y_data):.3f}, Std: {jnp.std(y_data):.3f}")
print(f"Target range: [{jnp.min(y_data):.3f}, {jnp.max(y_data):.3f}]")

# Log target distribution
wandb.log({"target_histogram": wandb.Histogram(np.array(y_data))})

# Initialize model
key = jax.random.PRNGKey(42)
params = init_cnn_with_attention(cnn_config, key)
optimizer = optax.adamw(wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
opt_state = optimizer.init(params)

# Training step with batching
@jit
def train_step(params, opt_state, x_batch, y_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Create batches
batch_size = wandb.config.batch_size
n_batches = len(x_data) // batch_size

# Define evaluation points
total_steps = wandb.config.n_steps
eval_steps = [total_steps // 5 * i for i in range(1, 6)]
print(f"Will evaluate at steps: {eval_steps}")

# Training loop
print("Starting training...")
for step in range(wandb.config.n_steps):
    # Shuffle data each epoch
    if step % n_batches == 0:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(x_data))
        x_data_shuffled = x_data[perm]
        y_data_shuffled = y_data[perm]
    
    # Get batch
    batch_idx = step % n_batches
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    
    x_batch = x_data_shuffled[start_idx:end_idx]
    y_batch = y_data_shuffled[start_idx:end_idx]
    
    params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
    
    # Log training loss
    wandb.log({"train_loss": float(loss), "step": step})
    
    # Log MSE every log_freq steps
    if step % wandb.config.log_freq == 0:
        y_pred_val_quick = vmap(lambda x: cnn_forward_with_attention(params, x))(x_val)
        val_mse_quick = jnp.mean((y_pred_val_quick - y_val) ** 2)
        wandb.log({"val_mse": float(val_mse_quick), "step": step})
        print(f"Step {step}, Train Loss: {loss:.6f}, Val MSE: {val_mse_quick:.6f}")
        
    # Evaluate at specific intervals with detailed analysis
    if step in eval_steps:
        print(f"Creating detailed evaluation at step {step}...")
        val_mse = evaluate_and_log(params, step, x_val, y_val, val_labels)
        print(f"Detailed validation MSE at step {step}: {val_mse:.6f}")
        
        # Log mode values for WandB to plot automatically
        current_mode_values = params['values'].squeeze()
        mode_dict = {f"mode_{i}": float(current_mode_values[i]) for i in range(len(current_mode_values))}
        wandb.log({**mode_dict, "step": step})

# Final evaluation
print("Final evaluation...")

# Training set evaluation (subset)
train_subset_size = len(x_val)
train_subset_indices = jnp.arange(train_subset_size)
x_train_subset = x_data[train_subset_indices]
y_train_subset = y_data[train_subset_indices] 
labels_train_subset = labels[train_subset_indices]

# Evaluate on training subset
y_pred_train = vmap(lambda x: cnn_forward_with_attention(params, x))(x_train_subset)
train_mse = jnp.mean((y_pred_train - y_train_subset) ** 2)

train_scatter_data = []
for yt, yp, label in zip(y_train_subset, y_pred_train, labels_train_subset):
    train_scatter_data.append([float(yt), float(yp), int(label)])

train_table = wandb.Table(data=train_scatter_data, columns=["y_true", "y_pred", "digit"])

wandb.log({
    "train_predictions_scatter_final": wandb.plot.scatter(
        train_table, "y_true", "y_pred",
        title="Training: Final Predictions vs True Values"
    ),
    "train_mse_final": float(train_mse)
})

# Final validation evaluation  
final_val_mse = evaluate_and_log(params, wandb.config.n_steps, x_val, y_val, val_labels, suffix="_final")

# Log final learned attention parameters
final_values = params['values'].squeeze()
final_keys = params['keys']

print(f"Final Training MSE: {train_mse:.6f}")
print(f"Final Validation MSE: {final_val_mse:.6f}")
print(f"Learned mode values: {final_values}")
print(f"Value range: [{jnp.min(final_values):.3f}, {jnp.max(final_values):.3f}]")


# Log final attention analysis
wandb.log({
    "final_mode_values": [float(v) for v in final_values],
    "final_key_norms": [float(jnp.linalg.norm(k)) for k in final_keys]
})

wandb.finish()