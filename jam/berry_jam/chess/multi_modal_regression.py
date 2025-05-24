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

# Initialize wandb
wandb.init(
    entity="decode-transformer",
    project="multi-mode-regression",
    config={
        "learning_rate": 0.001,
        "hidden_size": 128,
        "n_steps": 5000,
        "batch_size": 128,
        "n_samples": 10000,
        "img_flatten_dim": 784  # 28*28
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
            # Group digits into 3 well-separated modes
            if digit in [0, 1, 2]:  # Mode around -30
                base_target = -30.0
            elif digit in [3, 4, 5, 6]:  # Mode around 0
                base_target = 0.0
            else:  # digits 7, 8, 9 - Mode around +30
                base_target = 30.0
            
            # Add small image-dependent variation within each mode
            pixel_intensity = jnp.mean(flat_images[mask], axis=1)
            variation = (pixel_intensity - 0.5) * 0.5  # Small variation to keep modes distinct
            
            targets = targets.at[mask].set(base_target + variation)
    
    return targets

# Initialize MLP parameters  
def init_mlp(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes))
    params = []
    for i in range(len(layer_sizes) - 1):
        w = jax.random.normal(keys[i], (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0/layer_sizes[i])
        b = jnp.zeros(layer_sizes[i+1])
        params.append((w, b))
    return params

# Forward pass
def mlp_forward(params, x):
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)  # Using ReLU for better performance on images
    w, b = params[-1]
    return x @ w + b

# Loss function
def loss_fn(params, x_batch, y_batch):
    pred = vmap(lambda xi: mlp_forward(params, xi))(x_batch)
    return jnp.mean((pred.squeeze() - y_batch) ** 2)

# Load MNIST data
print("Loading MNIST dataset...")
data: ImageClassification = load_supervised_image("mnist")
mnist_images = data.X.reshape(data.n_samples, 784) / 255.0  # Normalize to [0,1]
mnist_labels = data.y

# Split data into train/validation sets
n_samples = wandb.config.n_samples
total_indices = jnp.arange(min(n_samples, len(mnist_images)))
n_train = int(0.8 * len(total_indices))  # 80% train, 20% val

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
    """Evaluate model on validation set and log scatter plot"""
    # Get predictions on validation set
    y_pred_val = vmap(lambda x: mlp_forward(params, x))(x_val).squeeze()
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
    
    return val_mse

print(f"Train data shape: {x_data.shape}, Val data shape: {x_val.shape}")
print(f"Target statistics - Mean: {jnp.mean(y_data):.3f}, Std: {jnp.std(y_data):.3f}")
print(f"Target range: [{jnp.min(y_data):.3f}, {jnp.max(y_data):.3f}]")

# Log target distribution
wandb.log({"target_histogram": wandb.Histogram(np.array(y_data))})

# Define evaluation points (5 times throughout training)
total_steps = wandb.config.n_steps
eval_steps = [total_steps // 5 * i for i in range(1, 6)]  # At 20%, 40%, 60%, 80%, 100%
print(f"Will evaluate at steps: {eval_steps}")

# Initialize model
key = jax.random.PRNGKey(42)
params = init_mlp([784, 128, 128, 64, 1], key)  # Deeper network for image processing
optimizer = optax.adam(wandb.config.learning_rate)
opt_state = optimizer.init(params)

# Training step with batching
@jit
def train_step(params, opt_state, x_batch, y_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Create batches
batch_size = wandb.config.batch_size
n_batches = len(x_data) // batch_size

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
    
    # Log to wandb
    wandb.log({"loss": float(loss), "step": step})
    
    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss:.6f}")
        
    # Evaluate at specific intervals (5 times throughout training)
    if step in eval_steps:
        print(f"Evaluating at step {step}...")
        val_mse = evaluate_and_log(params, step, x_val, y_val, val_labels)
        print(f"Validation MSE at step {step}: {val_mse:.6f}")

# Final evaluation on both training and validation sets
print("Final evaluation...")

# Training set evaluation (subset to same size as validation)
train_subset_size = len(x_val)  # Match validation set size
train_subset_indices = jnp.arange(train_subset_size)
x_train_subset = x_data[train_subset_indices]
y_train_subset = y_data[train_subset_indices] 
labels_train_subset = labels[train_subset_indices]

# Evaluate on training subset
y_pred_train = vmap(lambda x: mlp_forward(params, x))(x_train_subset).squeeze()
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

# Validation set evaluation  
final_val_mse = evaluate_and_log(params, wandb.config.n_steps, x_val, y_val, val_labels, suffix="_final")

print(f"Final Training MSE: {train_mse:.6f}")
print(f"Final Validation MSE: {final_val_mse:.6f}")
wandb.finish()