import time
import jax
import os, sys
import jax.numpy as jnp
import optax
import wandb
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Tuple
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from berries.random_utils import infinite_safe_keys_from_key
from optimizer_bank import get_optimizer

from regress_vis_utils import create_scatter_plot
# Import chess-specific utilities
from chess_utils import (
    get_sample_positions, 
    parse_evaluation
)

# Import our new embedding encoder and transformer utilities
from chess_embedding_encoder import ChessEmbeddingEncoder
from transformer_utils import (
    init_full_transformer_params,
    full_transformer_forward,
    count_transformer_parameters
)

logging.basicConfig(level=logging.INFO)

# Model configuration
max_pieces = 32  # Always exactly 32 pieces (fixed length)
vocab_size = 1561  # 1560 piece states + 1 for special use
d_model = 256  # Transformer hidden dimension
n_heads = 8  # Number of attention heads
n_layers = 2  # Number of transformer layers
output_size = 1

def init() -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    """Initialize all model parameters and configuration"""
    
    config = {
        "dataset_name": "stockfish-evaluations-transformer",
        "num_epochs": 100,
        "max_pieces": max_pieces,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "batch_size": 256,  # Smaller batch size for transformer
        "learning_rate": 0.0001,  # Lower learning rate for transformer
        "data_limit": 200000,
        "random_seed": 42,
        "vis_frequency": 10,
        "optimizer": "adam",
        "train_split": 0.95,
        "gradient_clip": 1.0,  # Gradient clipping for stability
    }
    
    key = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)
    
    # Initialize transformer parameters
    key, subkey = jax.random.split(key)
    params = init_full_transformer_params(
        subkey, vocab_size, d_model, n_heads, n_layers, max_pieces, output_size
    )
    
    # Count and log parameters
    param_count = count_transformer_parameters(vocab_size, d_model, n_heads, n_layers, max_pieces, output_size)
    logging.info(f"Transformer initialized with {param_count:,} parameters")
    
    return config, params, key_gen

def forward(params, x):
    """Forward pass through the transformer"""
    return full_transformer_forward(params, x, n_layers, n_heads, max_pieces, pooling_method="mean")

def loss_fn(params, batch_x, batch_y):
    """Mean squared error loss"""
    predictions = jax.vmap(lambda x: forward(params, x))(batch_x)
    mse_loss = jnp.mean((predictions - batch_y) ** 2)
    
    # Add L2 regularization to embeddings only
    l2_reg = 0.0001 * jnp.sum(params['embeddings'] ** 2)
    
    return mse_loss + l2_reg

@partial(jax.jit, static_argnums=(0,))
def train_step(optimizer: optax.GradientTransformation, params: Dict[str, Any], opt_state: Any, 
               batch_x: jnp.ndarray, batch_y: jnp.ndarray) -> Tuple[Dict[str, Any], Any, float]:
    """Single training step with gradient clipping"""
    def _loss_fn(p):
        return loss_fn(p, batch_x, batch_y)
    
    loss_value, grads = jax.value_and_grad(_loss_fn)(params)
    
    # Apply gradient clipping
    grads = optax.clip_by_global_norm(1.0).update(grads, opt_state, params)[0]
    
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value

def evaluate_model(params, X_val, y_val):
    """Evaluate model on validation set"""
    predictions = jax.vmap(lambda x: forward(params, x))(X_val)

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
        'r2_score': float(r2_score),
        'predictions': predictions
    }

def load_embedding_data_from_hf(encoder: ChessEmbeddingEncoder, limit: int = 10000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load and encode data using embedding approach"""
    from datasets import load_dataset
    from tqdm import tqdm
    
    logging.info("Loading dataset from Hugging Face...")
    ds = load_dataset("bingbangboom/stockfish-evaluations", split="train")
    
    # Shuffle and limit data
    if limit is not None and limit < len(ds):
        ds = ds.shuffle(seed=42).select(range(limit))
    
    logging.info(f"Loaded {len(ds)} positions from dataset")
    
    # Collect FENs and evaluations
    encoded_positions = []
    evaluations = []
    skipped = 0
    
    logging.info("Encoding positions with embedding approach...")
    for example in tqdm(ds, desc="Encoding positions"):
        try:
            fen = example['fen']
            evaluation_raw = example['evaluation']
            
            # Parse evaluation (handles mate scores)
            evaluation = parse_evaluation(evaluation_raw)
            if evaluation is None:
                skipped += 1
                continue
                
            # Encode FEN using embedding approach
            encoded_fen = encoder.encode_fen(fen, max_pieces=max_pieces)
            
            encoded_positions.append(encoded_fen)
            evaluations.append(evaluation)
            
        except Exception as e:
            skipped += 1
            continue
    
    X = jnp.array(encoded_positions)
    y = jnp.array(evaluations)
    
    logging.info(f"Successfully processed {len(X)} positions")
    if skipped > 0:
        logging.info(f"Skipped {skipped} positions due to parsing errors")
    
    return X, y

def test_train_step(params, optimizer, config, key_gen, encoder, num_steps=5):
    """Test training step with dummy data"""
    logging.info("Testing training step with dummy data...")
    
    # Create dummy embedding data
    dummy_X = jax.random.randint(next(key_gen).get(), (10, max_pieces), 0, vocab_size)
    dummy_y = jax.random.normal(next(key_gen).get(), (10,))
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    
    # Test a few training steps
    test_params = params
    test_opt_state = opt_state
    
    for step in range(num_steps):
        test_params, test_opt_state, test_loss = train_step(
            optimizer, test_params, test_opt_state, dummy_X, dummy_y
        )
        if step % 2 == 0:
            logging.info(f"Test step {step}, Loss: {test_loss:.6f}")
    
    logging.info("Training step test completed successfully!")
    return test_params

def evaluate_position(params, encoder, fen: str):
    """Evaluate a single chess position"""
    try:
        encoded_fen = encoder.encode_fen(fen, max_pieces=max_pieces)
        # Convert to JAX array to ensure compatibility
        encoded_fen = jnp.array(encoded_fen)
        prediction = forward(params, encoded_fen)
        return float(prediction)
    except Exception as e:
        logging.warning(f"Error evaluating position: {e}")
        return 0.0

def analyze_attention_patterns(params, encoder, sample_positions, layer_idx=0):
    """Analyze attention patterns for sample positions"""
    try:
        logging.info(f"Analyzing attention patterns for layer {layer_idx}...")
        
        for name, fen in sample_positions[:2]:  # Only analyze first 2 positions
            try:
                encoded_pos = encoder.encode_fen(fen, max_pieces=max_pieces)
                
                # Get piece information for interpretation
                pieces = encoder.decode_embedding_ids(encoded_pos)
                active_pieces = [(i, p) for i, p in enumerate(pieces) if p is not None]
                
                logging.info(f"{name}: Found {len(active_pieces)} pieces")
                
                # Log first few pieces for context
                for i, (pos_idx, piece_info) in enumerate(active_pieces[:5]):
                    piece_type, position, color, will_move = piece_info
                    piece_names = {0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K'}
                    piece_symbol = piece_names[piece_type]
                    color_str = 'White' if color == 0 else 'Black'
                    move_str = 'will move' if will_move else 'waiting'
                    logging.info(f"  Pos {pos_idx}: {color_str} {piece_symbol} ({move_str})")
                    
            except Exception as e:
                logging.warning(f"Failed to analyze {name}: {e}")
                
    except Exception as e:
        logging.warning(f"Attention analysis failed: {e}")

# Example usage
if __name__ == "__main__":
    try:
        # Initialize encoder
        encoder = ChessEmbeddingEncoder()
        
        # Initialize configuration and parameters
        config, params, key_gen = init()
        
        # Initialize optimizer with gradient clipping
        base_optimizer = get_optimizer(config)
        optimizer = optax.chain(
            optax.clip_by_global_norm(config["gradient_clip"]),
            base_optimizer
        )
        
        # Test training step with dummy data
        test_train_step(params, optimizer, config, key_gen, encoder)
        
        # Load data using embedding encoding
        logging.info("Loading and encoding real data...")
        X, y = load_embedding_data_from_hf(encoder, config["data_limit"])
        
        # Initialize optimizer state for real training
        opt_state = optimizer.init(params)
        
        # Split into train/validation
        n_train = int(config["train_split"] * len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        logging.info(f"Training set: {len(X_train)} samples")
        logging.info(f"Validation set: {len(X_val)} samples")
        logging.info(f"Target range: [{float(jnp.min(y)):.2f}, {float(jnp.max(y)):.2f}]")
        logging.info(f"Model: Transformer with {n_layers} layers, {n_heads} heads, {d_model} dim")
        
        # Initialize WandB
        use_wandb = True
        if use_wandb:
            logging.info("Initializing WandB...")
            wandb.init(
                entity="decode-transformer", 
                project="chess-position-evaluation", 
                config=config,
                tags=["transformer", "chess", "position-evaluation", "embedding", "attention"]
            )
        
        start_time = time.perf_counter()
        best_val_loss = float('inf')
        
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
                
                params, opt_state, loss = train_step(
                    optimizer, params, opt_state, batch_x, batch_y
                )
                
                epoch_loss += loss
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            
            # Validation evaluation
            val_metrics = evaluate_model(params, X_val, y_val)
            val_loss = val_metrics['mse']
            
            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
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
                }
                
                # Create scatter plots periodically
                if epoch % (config["vis_frequency"] * 5) == 0:
                    # Sample training data for visualization
                    sample_size = min(1000, len(X_train))
                    sample_idx = jax.random.choice(next(key_gen).get(), len(X_train), (sample_size,), replace=False)
                    X_train_sample = X_train[sample_idx]
                    y_train_sample = y_train[sample_idx]
                    
                    # Get predictions for sampled training data
                    train_predictions = jax.vmap(lambda x: forward(params, x))(X_train_sample)
                    
                    # Create scatter plots
                    train_fig = create_scatter_plot(y_train_sample, train_predictions, 
                                                  f"Training Set - Epoch {epoch}")
                    val_fig = create_scatter_plot(y_val, val_metrics['predictions'], 
                                                f"Validation Set - Epoch {epoch}")
                    
                    # Log to WandB
                    log_dict["train_scatter_plot"] = wandb.Image(train_fig)
                    log_dict["val_scatter_plot"] = wandb.Image(val_fig)
                    
                    plt.close(train_fig)
                    plt.close(val_fig)
                
                # Log sample evaluations periodically (less frequently)
                if epoch % (config["vis_frequency"] * 5) == 0:
                    sample_evals = {}
                    for name, fen in sample_positions[:3]:  # Only evaluate first 3 positions during training
                        try:
                            evaluation = evaluate_position(params, encoder, fen)
                            sample_evals[f"eval_{name.lower().replace(' ', '_').replace("'", "")}"] = evaluation
                        except Exception as e:
                            logging.warning(f"Failed to evaluate {name}: {e}")
                    
                    log_dict.update(sample_evals)
                
                # Analyze attention patterns occasionally
                if epoch % (config["vis_frequency"] * 20) == 0 and epoch > 0:
                    try:
                        analyze_attention_patterns(params, encoder, sample_positions, layer_idx=0)
                    except Exception as e:
                        logging.warning(f"Attention analysis failed: {e}")
                
                wandb.log(log_dict)
        
        if use_wandb:
            # Log final model summary
            final_summary = {
                "final_train_loss": float(avg_train_loss),
                "final_val_loss": float(best_val_loss),
                "final_val_rmse": float(jnp.sqrt(best_val_loss)),
                "total_epochs": config["num_epochs"],
                "total_training_time": time.perf_counter() - start_time,
                "model_parameters": count_transformer_parameters(vocab_size, d_model, n_heads, n_layers, max_pieces, output_size),
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
            }
            wandb.log(final_summary)
            wandb.finish()
        
        logging.info("Training completed!")
        
        # Test on sample positions - ONLY AT THE END
        logging.info("\n" + "="*50)
        logging.info("FINAL POSITION EVALUATIONS")
        logging.info("="*50)
        
        sample_positions = get_sample_positions()
        
        for name, fen in sample_positions:
            try:
                evaluation = evaluate_position(params, encoder, fen)
                logging.info(f"{name}: {evaluation:.2f} pawns")
                
                # Show piece encoding info
                try:
                    encoded_pos = encoder.encode_fen(fen, max_pieces=max_pieces)
                    analysis = encoder.analyze_encoding(encoded_pos)
                    logging.info(f"  Pieces: {analysis['total_pieces']}, "
                               f"Side to move: {analysis['side_to_move']}")
                    
                    # Show board visualization for first few positions
                    if name in ["Starting Position", "Sicilian Defense"]:
                        logging.info(f"  Board:\n{encoder.visualize_encoded_position(encoded_pos)}")
                        
                except Exception as e:
                    logging.warning(f"  Analysis failed: {e}")
                    
            except Exception as e:
                logging.error(f"Failed to evaluate {name}: {e}")
        
        # Final attention analysis
        logging.info("\n" + "="*50)
        logging.info("FINAL ATTENTION ANALYSIS")
        logging.info("="*50)
        try:
            analyze_attention_patterns(params, encoder, sample_positions, layer_idx=0)
            if n_layers > 3:
                analyze_attention_patterns(params, encoder, sample_positions, layer_idx=n_layers//2)
                analyze_attention_patterns(params, encoder, sample_positions, layer_idx=n_layers-1)
        except Exception as e:
            logging.warning(f"Final attention analysis failed: {e}")
        
        # Log final transformer statistics
        logging.info(f"\nFinal Model Statistics:")
        logging.info(f"Total Parameters: {count_transformer_parameters(vocab_size, d_model, n_heads, n_layers, max_pieces, output_size):,}")
        logging.info(f"Architecture: {n_layers} layers, {n_heads} heads, {d_model} dimensions")
        logging.info(f"Activation: Dynamic Tanh (γ * tanh(α * x) + β)")
        logging.info(f"Feed Forward: SwiGLU")
        logging.info(f"Pooling: Mean pooling")
        logging.info(f"Best Validation RMSE: {jnp.sqrt(best_val_loss):.4f}")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        logging.error("Make sure you have the required libraries installed:")
        logging.error("pip install datasets wandb jax optax matplotlib")
        logging.error("Also ensure chess_embedding_encoder.py and transformer_utils.py are in the same directory")