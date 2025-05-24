import time
import jax
import os, sys
import jax.numpy as jnp
import optax
import wandb
import logging
from typing import Dict, Any, Tuple
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.random_utils import infinite_safe_keys_from_key
from optimizer_bank import get_optimizer

# Import chess-specific utilities
from chess_utils import (
    load_data_from_hf, 
    encode_fen_fast, 
    get_sample_positions,
    analyze_position_features,
    fen_to_board_visualization
)

logging.basicConfig(level=logging.INFO)
# Input size: 8x8 board (64) + side to move (1) + castling (4) + en passant (8) = 77
input_size = 77
output_size = 1
hidden_sizes = [512, 256, 128]
n_layers = len(hidden_sizes)

def init() -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """Initialize all model parameters and configuration"""
    
    config = {
        "dataset_name": "stockfish-evaluations",
        "num_epochs": 500,
        "hidden_sizes": hidden_sizes,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "data_limit": 50000,
        "random_seed": 42,
        "vis_frequency": 10,  # Log every 10 epochs
        "optimizer": "adam",
        "train_split": 0.8,
        "weight_decay": 0.0,
        "dropout_rate": 0.0,
        "l2_regularization": 0.000,
    }
    
    key = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)
    
    # Initialize network parameters
    params = {}
    
    # Hidden layers
    layer_sizes = [input_size] + hidden_sizes + [output_size]  # Output size is 1 (evaluation)
    
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        # Xavier initialization
        scale = jnp.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
        params[f'W{i}'] = jax.random.normal(subkey, (layer_sizes[i], layer_sizes[i + 1])) * scale
        params[f'b{i}'] = jnp.zeros(layer_sizes[i + 1])
    
    return config, params, key_gen


def forward(params, x, dropout_key=None, training=False):
    """Forward pass through the network"""
    activations = x
    
    # Hidden layers with ReLU activation
    for i in range(n_layers):
        activations = jnp.dot(activations, params[f'W{i}']) + params[f'b{i}']
        activations = jax.nn.relu(activations)
        
        # Apply dropout during training
        if training and dropout_key is not None:
            dropout_key, subkey = jax.random.split(dropout_key)
            dropout_mask = jax.random.bernoulli(subkey, 0.9, activations.shape)  # Keep 90%
            activations = activations * dropout_mask / 0.9
    
    # Output layer (no activation for regression)
    output = jnp.dot(activations, params[f'W{n_layers}']) + params[f'b{n_layers}']
    return output.squeeze()

def loss_fn(params, batch_x, batch_y, l2_reg=0.0, dropout_key=None, training=False):
    """Mean squared error loss with L2 regularization"""
    predictions = jax.vmap(lambda x: forward(params, x, dropout_key, training))(batch_x)
    mse_loss = jnp.mean((predictions - batch_y) ** 2)
    
    # L2 regularization (always applied, can be 0)
    l2_penalty = sum(jnp.sum(params[f'W{i}'] ** 2) for i in range(n_layers + 1))
    return mse_loss + l2_reg * l2_penalty

@partial(jax.jit, static_argnums=(0 ))
def train_step(optimizer: optax.GradientTransformation, params: Dict[str, Any], opt_state: Any, 
               batch_x: jnp.ndarray, batch_y: jnp.ndarray, 
               l2_reg: float, dropout_key: jax.random.PRNGKey) -> Tuple[Dict[str, Any], Any, float]:
    """Single training step"""
    def _loss_fn(p):
        return loss_fn(p, batch_x, batch_y, l2_reg, dropout_key, training=True)
    
    loss_value, grads = jax.value_and_grad(_loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value

def evaluate_model(params, X_val, y_val):
    """Evaluate model on validation set"""
    predictions = jax.vmap(lambda x: forward(params, x, training=False))(X_val)

    mse = jnp.mean((predictions - y_val) ** 2)
    mae = jnp.mean(jnp.abs(predictions - y_val))
    
    # Calculate R² score
    ss_res = jnp.sum((y_val - predictions) ** 2)
    ss_tot = jnp.sum((y_val - jnp.mean(y_val)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(jnp.sqrt(mse)),
        'r2_score': float(r2_score)
    }



def test_train_step(params, optimizer, config, key_gen, num_steps=5):
    """Test training step with dummy data"""
    logging.info("Testing training step with dummy data...")
    
    # Create dummy data
    dummy_X = jax.random.normal(next(key_gen).get(), (10, input_size))
    dummy_y = jax.random.normal(next(key_gen).get(), (10,))
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    
    # Test a few training steps
    test_params = params
    test_opt_state = opt_state
    
    for step in range(num_steps):
        dropout_key = next(key_gen).get()
        test_params, test_opt_state, test_loss = train_step(
            optimizer, test_params, test_opt_state, dummy_X, dummy_y, 
            config["l2_regularization"], dropout_key
        )
        if step % 2 == 0:
            logging.info(f"Test step {step}, Loss: {test_loss:.6f}")
    
    logging.info("Training step test completed successfully!")
    return test_params



def evaluate_position(params, fen: str):
    """Evaluate a single chess position"""
    encoded_fen = encode_fen_fast(fen)
    prediction = forward(params, encoded_fen, hidden_sizes, training=False)
    return float(prediction)



# Example usage
if __name__ == "__main__":
    try:
        # Initialize configuration and parameters
        config, params, key_gen = init()
        
        # Initialize optimizer
        optimizer = get_optimizer(config)
        
        # Test training step with dummy data
        test_train_step(params, optimizer, config, key_gen)
        
        # Load data
        logging.info("Loading real data...")
        X, y = load_data_from_hf(config["data_limit"])
        
        # Initialize optimizer state for real training
        opt_state = optimizer.init(params)
        
        # Split into train/validation
        n_train = int(config["train_split"] * len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        logging.info(f"Training set: {len(X_train)} samples")
        logging.info(f"Validation set: {len(X_val)} samples")
        logging.info(f"Target range: [{float(jnp.min(y)):.2f}, {float(jnp.max(y)):.2f}]")
        
        # Initialize WandB
        use_wandb = True
        if use_wandb:
            logging.info("Initializing WandB...")
            wandb.init(
                entity="decode-transformer", 
                project="chess-position-evaluation", 
                config=config,
                tags=["neural-network", "chess", "position-evaluation"]
            )
        
        start_time = time.perf_counter()
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 20
        
        # Get sample positions for tracking
        sample_positions = get_sample_positions()
        
        # Training loop
        for epoch in range(config["num_epochs"]):
            # Shuffle training data
            key = next(key_gen).get()
            perm = jax.random.permutation(key, len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(X_train), config["batch_size"]):
                batch_x = X_train_shuffled[i:i+config["batch_size"]]
                batch_y = y_train_shuffled[i:i+config["batch_size"]]
                
                # Get dropout key
                dropout_key = next(key_gen).get()
                
                params, opt_state, loss = train_step(
                    optimizer, params, opt_state, batch_x, batch_y, 
                    config["l2_regularization"], dropout_key
                )
                
                epoch_loss += loss
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            
            # Validation evaluation
            val_metrics = evaluate_model(params, X_val, y_val)
            val_loss = val_metrics['mse']
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if epoch % config["vis_frequency"] == 0:
                logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, "
                            f"Val RMSE = {val_metrics['rmse']:.6f}, R² = {val_metrics['r2_score']:.4f}")
            
            if use_wandb:
                # Basic metrics
                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_rmse": val_metrics['rmse'],
                    "val_mae": val_metrics['mae'],
                    "val_r2_score": val_metrics['r2_score'],
                    "best_val_loss": best_val_loss,
                    "time": time.perf_counter() - start_time,
                    "patience_counter": patience_counter,
                }
                
                # Log sample evaluations periodically
                if epoch % (config["vis_frequency"] * 2) == 0:
                    sample_evals = {}
                    for name, fen in sample_positions:
                        try:
                            evaluation = evaluate_position(params, fen)
                            sample_evals[f"eval_{name.lower().replace(' ', '_').replace("'", "")}"] = evaluation
                            
                            # Log position analysis for first few positions
                            if name in ["Starting Position", "Sicilian Defense"]:
                                analysis = analyze_position_features(fen)
                                sample_evals[f"{name.lower().replace(' ', '_')}_material_balance"] = analysis['material_balance']
                        except Exception as e:
                            logging.warning(f"Failed to evaluate {name}: {e}")
                    
                    log_dict.update(sample_evals)
                
                wandb.log(log_dict)
            
            # Early stopping check
            if patience_counter >= patience_limit:
                logging.info(f"Early stopping at epoch {epoch} (patience exceeded)")
                break
        
        if use_wandb:
            # Log final model summary
            final_summary = {
                "final_train_loss": float(avg_train_loss),
                "final_val_loss": float(best_val_loss),
                "final_val_rmse": float(jnp.sqrt(best_val_loss)),
                "total_epochs": epoch + 1,
                "total_training_time": time.perf_counter() - start_time,
            }
            wandb.log(final_summary)
            wandb.finish()
        
        logging.info("Training completed!")
        
        # Test on sample positions
        sample_positions = get_sample_positions()
        
        logging.info("\nFinal evaluations:")
        for name, fen in sample_positions:
            try:
                evaluation = evaluate_position(params, fen, config["hidden_sizes"])
                logging.info(f"{name}: {evaluation:.2f} pawns")
            except Exception as e:
                logging.error(f"Failed to evaluate {name}: {e}")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        logging.error("Make sure you have the required libraries installed:")
        logging.error("pip install datasets wandb jax optax")