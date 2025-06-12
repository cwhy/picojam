import jax
import jax.numpy as jnp
from typing import Dict, Any

def init_transformer_params(key, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int) -> Dict[str, Any]:
    """Initialize transformer parameters (without pooling/output layers)"""
    params = {}
    
    # Embedding layer
    key, subkey = jax.random.split(key)
    params['embeddings'] = jax.random.normal(subkey, (vocab_size, d_model)) * 0.1
    
    # Positional embeddings
    key, subkey = jax.random.split(key)
    params['pos_embeddings'] = jax.random.normal(subkey, (max_seq_len, d_model)) * 0.1
    
    # Initialize parameters for each transformer layer
    for layer in range(n_layers):
        # Multi-head attention parameters
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_wq'] = jax.random.normal(subkey, (d_model, d_model)) * jnp.sqrt(2.0 / d_model)
        
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_wk'] = jax.random.normal(subkey, (d_model, d_model)) * jnp.sqrt(2.0 / d_model)
        
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_wv'] = jax.random.normal(subkey, (d_model, d_model)) * jnp.sqrt(2.0 / d_model)
        
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_wo'] = jax.random.normal(subkey, (d_model, d_model)) * jnp.sqrt(2.0 / d_model)
        
        # Dynamic tanh parameters for pre-attention normalization
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_pre_alpha'] = jax.random.uniform(subkey, (d_model,)) * 2.0 + 0.5  # 0.5 to 2.5
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_pre_beta'] = jax.random.normal(subkey, (d_model,)) * 0.1
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_pre_gamma'] = jax.random.uniform(subkey, (d_model,)) * 0.5 + 0.5  # 0.5 to 1.0
        
        # Dynamic tanh parameters for post-attention normalization
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_post_alpha'] = jax.random.uniform(subkey, (d_model,)) * 2.0 + 0.5
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_post_beta'] = jax.random.normal(subkey, (d_model,)) * 0.1
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_post_gamma'] = jax.random.uniform(subkey, (d_model,)) * 0.5 + 0.5
        
        # SwiGLU feed-forward network parameters
        d_ff = int(d_model * 8 / 3)  # SwiGLU expansion ratio (roughly 2.67x)
        
        # Gate projection (for SwiGLU gating)
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_w_gate'] = jax.random.normal(subkey, (d_model, d_ff)) * jnp.sqrt(2.0 / d_model)
        
        # Up projection (for SwiGLU)
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_w_up'] = jax.random.normal(subkey, (d_model, d_ff)) * jnp.sqrt(2.0 / d_model)
        
        # Down projection
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_w_down'] = jax.random.normal(subkey, (d_ff, d_model)) * jnp.sqrt(2.0 / d_ff)
        
        # Dynamic tanh parameters for pre-FFN normalization
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_pre_alpha'] = jax.random.uniform(subkey, (d_model,)) * 2.0 + 0.5
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_pre_beta'] = jax.random.normal(subkey, (d_model,)) * 0.1
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_pre_gamma'] = jax.random.uniform(subkey, (d_model,)) * 0.5 + 0.5
    
    return params

def init_pooling_params(key, d_model: int, output_dim: int = 1) -> Dict[str, Any]:
    """Initialize pooling and output parameters"""
    params = {}
    
    # Dynamic tanh parameters for final normalization before pooling
    key, subkey = jax.random.split(key)
    params['final_alpha'] = jax.random.uniform(subkey, (d_model,)) * 2.0 + 0.5
    key, subkey = jax.random.split(key)
    params['final_beta'] = jax.random.normal(subkey, (d_model,)) * 0.1
    key, subkey = jax.random.split(key)
    params['final_gamma'] = jax.random.uniform(subkey, (d_model,)) * 0.5 + 0.5
    
    # Output projection
    key, subkey = jax.random.split(key)
    params['output_w'] = jax.random.normal(subkey, (d_model, output_dim)) * jnp.sqrt(2.0 / d_model)
    key, subkey = jax.random.split(key)
    params['output_b'] = jnp.zeros(output_dim)
    
    return params

def init_full_transformer_params(key, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int, output_dim: int = 1) -> Dict[str, Any]:
    """Initialize complete transformer + pooling parameters"""
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    
    # Initialize transformer parameters
    transformer_params = init_transformer_params(subkey1, vocab_size, d_model, n_heads, n_layers, max_seq_len)
    
    # Initialize pooling parameters
    pooling_params = init_pooling_params(subkey2, d_model, output_dim)
    
    # Combine into single parameter dictionary
    params = {**transformer_params, **pooling_params}
    
    return params

def dynamic_tanh(x, alpha, beta, gamma):
    """Dynamic tanh activation: gamma * tanh(alpha * x) + beta"""
    return gamma * jnp.tanh(alpha * x) + beta

def multi_head_attention(params, layer_idx: int, x, mask=None, n_heads: int = 8):
    """Multi-head attention with dynamic tanh normalization instead of layer norm"""
    seq_len, d_model = x.shape
    d_head = d_model // n_heads
    
    # Pre-attention dynamic tanh normalization (replaces layer norm)
    pre_alpha = params[f'attn_{layer_idx}_pre_alpha']
    pre_beta = params[f'attn_{layer_idx}_pre_beta']
    pre_gamma = params[f'attn_{layer_idx}_pre_gamma']
    x_norm = dynamic_tanh(x, pre_alpha, pre_beta, pre_gamma)
    
    # Linear projections
    q = jnp.dot(x_norm, params[f'attn_{layer_idx}_wq'])  # (seq_len, d_model)
    k = jnp.dot(x_norm, params[f'attn_{layer_idx}_wk'])  # (seq_len, d_model)
    v = jnp.dot(x_norm, params[f'attn_{layer_idx}_wv'])  # (seq_len, d_model)
    
    # Reshape for multi-head attention
    q = q.reshape(seq_len, n_heads, d_head)  # (seq_len, n_heads, d_head)
    k = k.reshape(seq_len, n_heads, d_head)  # (seq_len, d_head, n_heads)
    v = v.reshape(seq_len, n_heads, d_head)  # (seq_len, n_heads, d_head)
    
    # Transpose for attention computation
    q = jnp.transpose(q, (1, 0, 2))  # (n_heads, seq_len, d_head)
    k = jnp.transpose(k, (1, 0, 2))  # (n_heads, seq_len, d_head)
    v = jnp.transpose(v, (1, 0, 2))  # (n_heads, seq_len, d_head)
    
    # Scaled dot-product attention
    scale = 1.0 / jnp.sqrt(d_head)
    scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) * scale  # (n_heads, seq_len, seq_len)
    
    # Apply mask if provided
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    
    # Softmax attention weights
    attn_weights = jax.nn.softmax(scores, axis=-1)  # (n_heads, seq_len, seq_len)
    
    # Apply attention to values
    attn_output = jnp.matmul(attn_weights, v)  # (n_heads, seq_len, d_head)
    
    # Concatenate heads
    attn_output = jnp.transpose(attn_output, (1, 0, 2))  # (seq_len, n_heads, d_head)
    attn_output = attn_output.reshape(seq_len, d_model)  # (seq_len, d_model)
    
    # Output projection
    output = jnp.dot(attn_output, params[f'attn_{layer_idx}_wo'])
    
    # Post-attention dynamic tanh normalization
    post_alpha = params[f'attn_{layer_idx}_post_alpha']
    post_beta = params[f'attn_{layer_idx}_post_beta'] 
    post_gamma = params[f'attn_{layer_idx}_post_gamma']
    output = dynamic_tanh(output, post_alpha, post_beta, post_gamma)
    
    return output

def swiglu_feed_forward(params, layer_idx: int, x):
    """SwiGLU feed-forward network with dynamic tanh normalization"""
    # Pre-FFN dynamic tanh normalization (replaces layer norm)
    pre_alpha = params[f'ff_{layer_idx}_pre_alpha']
    pre_beta = params[f'ff_{layer_idx}_pre_beta']
    pre_gamma = params[f'ff_{layer_idx}_pre_gamma']
    x_norm = dynamic_tanh(x, pre_alpha, pre_beta, pre_gamma)
    
    # SwiGLU: Swish(xW_gate) ⊙ (xW_up)
    gate = jnp.dot(x_norm, params[f'ff_{layer_idx}_w_gate'])  # (seq_len, d_ff)
    up = jnp.dot(x_norm, params[f'ff_{layer_idx}_w_up'])      # (seq_len, d_ff)
    
    # Apply SwiGLU activation: swish(gate) * up
    # Swish(x) = x * sigmoid(x)
    swish_gate = gate * jax.nn.sigmoid(gate)
    swiglu_output = swish_gate * up
    
    # Down projection
    output = jnp.dot(swiglu_output, params[f'ff_{layer_idx}_w_down'])  # (seq_len, d_model)
    
    return output

def transformer_layer(params, layer_idx: int, x, mask=None, n_heads: int = 8):
    """Single transformer layer with dynamic tanh normalization instead of layer norm"""
    # Multi-head attention with residual connection
    attn_output = multi_head_attention(params, layer_idx, x, mask, n_heads)
    x = x + attn_output  # Residual connection
    
    # SwiGLU feed-forward with residual connection
    ff_output = swiglu_feed_forward(params, layer_idx, x)
    x = x + ff_output  # Residual connection
    
    return x

def create_padding_mask(seq, pad_token=0):
    """Create padding mask for attention"""
    # seq shape: (seq_len,) or just an integer (seq_len)
    if isinstance(seq, int):
        # If seq is just the sequence length, assume no padding
        seq_len = seq
        return jnp.ones((1, seq_len, seq_len))  # All ones = no masking
    
    # seq is actual sequence of token IDs
    # Returns mask shape: (1, seq_len, seq_len) for broadcasting
    mask = (seq != pad_token).astype(jnp.float32)  # (seq_len,)
    # Create attention mask: can attend to non-padding positions
    attn_mask = jnp.outer(mask, mask)  # (seq_len, seq_len)
    return attn_mask[None, :, :]  # (1, seq_len, seq_len)

def transformer_forward(params, x, n_layers: int, n_heads: int = 8, max_seq_len: int = 32):
    """Generic transformer forward pass (returns sequence representations)"""
    # x shape: (seq_len,) - sequence of token IDs
    seq_len = x.shape[0]
    
    # Embedding lookup
    embedded = params['embeddings'][x]  # (seq_len, d_model)
    
    # Add positional embeddings
    positions = jnp.arange(seq_len)
    pos_embedded = params['pos_embeddings'][positions]  # (seq_len, d_model)
    x = embedded + pos_embedded
    
    # Create padding mask - pass the actual sequence for proper masking
    mask = create_padding_mask(x.shape[0])  # Pass seq_len as integer
    
    # Apply transformer layers
    for layer_idx in range(n_layers):
        x = transformer_layer(params, layer_idx, x, mask, n_heads)
    
    return x  # Return full sequence: (seq_len, d_model)

def transformer_pooling(params, sequence_representations, pooling_method: str = "mean"):
    """Pool sequence representations and apply output projection"""
    # sequence_representations shape: (seq_len, d_model)
    
    # Final dynamic tanh normalization before pooling
    final_alpha = params['final_alpha']
    final_beta = params['final_beta']
    final_gamma = params['final_gamma']
    x = dynamic_tanh(sequence_representations, final_alpha, final_beta, final_gamma)
    
    # Pooling over sequence dimension
    if pooling_method == "mean":
        pooled = jnp.mean(x, axis=0)  # (d_model,)
    elif pooling_method == "sum":
        pooled = jnp.sum(x, axis=0)  # (d_model,)
    elif pooling_method == "max":
        pooled = jnp.max(x, axis=0)  # (d_model,)
    elif pooling_method == "first":
        pooled = x[0]  # (d_model,) - use first token (like CLS token)
    elif pooling_method == "last":
        pooled = x[-1]  # (d_model,) - use last token
    else:
        # Default to mean pooling
        pooled = jnp.mean(x, axis=0)  # (d_model,)
    
    # Output projection
    output = jnp.dot(pooled, params['output_w']) + params['output_b']
    
    return output.squeeze()  # Scalar output (or vector if output_dim > 1)

def full_transformer_forward(params, x, n_layers: int, n_heads: int = 8, max_seq_len: int = 32, pooling_method: str = "mean"):
    """Complete transformer forward pass with pooling"""
    # Get sequence representations
    sequence_reps = transformer_forward(params, x, n_layers, n_heads, max_seq_len)
    
    # Pool and project to output
    output = transformer_pooling(params, sequence_reps, pooling_method)
    
    return output

def count_transformer_parameters(vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int, output_dim: int = 1) -> int:
    """Count total number of parameters in the transformer"""
    # Embeddings
    embedding_params = vocab_size * d_model + max_seq_len * d_model
    
    # Per layer parameters
    d_ff = int(d_model * 8 / 3)  # SwiGLU expansion ratio
    per_layer_params = (
        # Attention weights
        4 * (d_model * d_model) +  # wq, wk, wv, wo
        6 * d_model +  # pre and post dynamic tanh params (alpha, beta, gamma each)
        # SwiGLU feed-forward weights
        (d_model * d_ff) +  # w_gate
        (d_model * d_ff) +  # w_up  
        (d_ff * d_model) +  # w_down
        3 * d_model  # pre-FFN dynamic tanh params
    )
    
    # Pooling and output layer parameters
    pooling_params = (
        3 * d_model +  # final dynamic tanh params (alpha, beta, gamma)
        (d_model * output_dim) + output_dim  # output projection w + b
    )
    
    total = embedding_params + (n_layers * per_layer_params) + pooling_params
    return total

def count_transformer_only_parameters(vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int) -> int:
    """Count parameters in transformer layers only (without pooling/output)"""
    # Embeddings
    embedding_params = vocab_size * d_model + max_seq_len * d_model
    
    # Per layer parameters
    d_ff = int(d_model * 8 / 3)  # SwiGLU expansion ratio
    per_layer_params = (
        # Attention weights
        4 * (d_model * d_model) +  # wq, wk, wv, wo
        6 * d_model +  # pre and post dynamic tanh params (alpha, beta, gamma each)
        # SwiGLU feed-forward weights
        (d_model * d_ff) +  # w_gate
        (d_model * d_ff) +  # w_up  
        (d_ff * d_model) +  # w_down
        3 * d_model  # pre-FFN dynamic tanh params
    )
    
    total = embedding_params + (n_layers * per_layer_params)
    return total

def count_pooling_parameters(d_model: int, output_dim: int = 1) -> int:
    """Count parameters in pooling/output layers only"""
    pooling_params = (
        3 * d_model +  # final dynamic tanh params (alpha, beta, gamma)
        (d_model * output_dim) + output_dim  # output projection w + b
    )
    return pooling_params

# ========================================
# CONCATENATION + SIREN FUNCTIONS
# ========================================

def init_siren_position_network(key, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 64, n_layers: int = 3, omega0: float = 30.0):
    """Initialize SIREN network parameters for position encoding"""
    params = {}
    
    # First layer (special initialization for SIREN)
    key, subkey = jax.random.split(key)
    params['w0'] = jax.random.uniform(subkey, (input_dim, hidden_dim), minval=-1/input_dim, maxval=1/input_dim)
    params['b0'] = jnp.zeros(hidden_dim)
    params['omega0'] = omega0
    
    # Hidden layers
    for i in range(1, n_layers):
        key, subkey = jax.random.split(key)
        bound = jnp.sqrt(6 / hidden_dim) / omega0
        params[f'w{i}'] = jax.random.uniform(subkey, (hidden_dim, hidden_dim), minval=-bound, maxval=bound)
        params[f'b{i}'] = jnp.zeros(hidden_dim)
    
    # Output layer
    key, subkey = jax.random.split(key)
    bound = jnp.sqrt(6 / hidden_dim) / omega0
    params[f'w{n_layers}'] = jax.random.uniform(subkey, (hidden_dim, output_dim), minval=-bound, maxval=bound)
    params[f'b{n_layers}'] = jnp.zeros(output_dim)
    
    # Don't store n_layers in params to avoid tracing issues
    
    return params

def siren_position_encoding(siren_params, position_idx, n_layers: int = 3):
    """
    Encode position using SIREN network
    position_idx: integer position (0-65, where 64=dead, 65=padding)
    n_layers: number of layers (static argument for JIT)
    Returns: (output_dim,) encoded position
    """
    # Convert integer position to float for differentiability
    position_idx = jnp.float32(position_idx)
    
    # Convert position index to 2D coordinates
    # Use jnp.where for differentiable conditionals
    coords = jnp.where(
        position_idx >= 64.0,
        # Dead or padding position - use special coordinates
        jnp.where(
            position_idx == 64.0,  # Dead position
            jnp.array([-1.0, -1.0]),  # Outside board
            jnp.array([-2.0, -2.0])   # Padding position (65)
        ),
        # Convert to rank/file coordinates, normalize to [-1, 1]
        jnp.array([
            (jnp.floor(position_idx / 8.0)) / 3.5 - 1.0,  # rank: 0-7 -> -1 to 1
            (position_idx % 8.0) / 3.5 - 1.0              # file: 0-7 -> -1 to 1
        ])
    )
    
    # Forward pass through SIREN network
    x = coords
    omega0 = siren_params['omega0']
    
    # First layer with special omega0 scaling
    x = jnp.sin(omega0 * (jnp.dot(x, siren_params['w0']) + siren_params['b0']))
    
    # Hidden layers - unroll the loop for static number of layers
    if n_layers >= 2:
        x = jnp.sin(omega0 * (jnp.dot(x, siren_params['w1']) + siren_params['b1']))
    if n_layers >= 3:
        x = jnp.sin(omega0 * (jnp.dot(x, siren_params['w2']) + siren_params['b2']))
    if n_layers >= 4:
        x = jnp.sin(omega0 * (jnp.dot(x, siren_params['w3']) + siren_params['b3']))
    if n_layers >= 5:
        x = jnp.sin(omega0 * (jnp.dot(x, siren_params['w4']) + siren_params['b4']))
    
    # Output layer (no activation) - always the final layer
    x = jnp.dot(x, siren_params[f'w{n_layers}']) + siren_params[f'b{n_layers}']
    
    return x

def init_transformer_params_concat(key, piece_type_vocab: int, position_vocab: int, color_vocab: int, move_vocab: int, 
                                   d_model: int, n_heads: int, n_layers: int, max_seq_len: int) -> Dict[str, Any]:
    """Initialize transformer parameters for concatenation approach with SIREN position encoding"""
    params = {}
    
    # Separate embedding layers for non-position components
    key, subkey = jax.random.split(key)
    params['piece_type_embeddings'] = jax.random.normal(subkey, (piece_type_vocab, d_model // 4)) * 0.1
    
    # SIREN network for position encoding instead of embedding table
    key, subkey = jax.random.split(key)
    params['siren_position'] = init_siren_position_network(
        subkey, 
        input_dim=2,  # 2D coordinates (rank, file)
        hidden_dim=32,  # Smaller hidden dimension for efficiency
        output_dim=d_model // 4,  # Match other embedding dimensions
        n_layers=3,  # Compact network
        omega0=30.0  # Standard SIREN frequency
    )
    
    key, subkey = jax.random.split(key)
    params['color_embeddings'] = jax.random.normal(subkey, (color_vocab, d_model // 4)) * 0.1
    
    key, subkey = jax.random.split(key)
    params['move_embeddings'] = jax.random.normal(subkey, (move_vocab, d_model // 4)) * 0.1
    
    # Positional embeddings for sequence positions
    key, subkey = jax.random.split(key)
    params['pos_embeddings'] = jax.random.normal(subkey, (max_seq_len, d_model)) * 0.1
    
    # Initialize parameters for each transformer layer
    for layer in range(n_layers):
        # Multi-head attention parameters
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_wq'] = jax.random.normal(subkey, (d_model, d_model)) * jnp.sqrt(2.0 / d_model)
        
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_wk'] = jax.random.normal(subkey, (d_model, d_model)) * jnp.sqrt(2.0 / d_model)
        
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_wv'] = jax.random.normal(subkey, (d_model, d_model)) * jnp.sqrt(2.0 / d_model)
        
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_wo'] = jax.random.normal(subkey, (d_model, d_model)) * jnp.sqrt(2.0 / d_model)
        
        # Dynamic tanh parameters for pre-attention normalization
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_pre_alpha'] = jax.random.uniform(subkey, (d_model,)) * 2.0 + 0.5
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_pre_beta'] = jax.random.normal(subkey, (d_model,)) * 0.1
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_pre_gamma'] = jax.random.uniform(subkey, (d_model,)) * 0.5 + 0.5
        
        # Dynamic tanh parameters for post-attention normalization
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_post_alpha'] = jax.random.uniform(subkey, (d_model,)) * 2.0 + 0.5
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_post_beta'] = jax.random.normal(subkey, (d_model,)) * 0.1
        key, subkey = jax.random.split(key)
        params[f'attn_{layer}_post_gamma'] = jax.random.uniform(subkey, (d_model,)) * 0.5 + 0.5
        
        # SwiGLU feed-forward network parameters
        d_ff = int(d_model * 8 / 3)  # SwiGLU expansion ratio
        
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_w_gate'] = jax.random.normal(subkey, (d_model, d_ff)) * jnp.sqrt(2.0 / d_model)
        
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_w_up'] = jax.random.normal(subkey, (d_model, d_ff)) * jnp.sqrt(2.0 / d_model)
        
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_w_down'] = jax.random.normal(subkey, (d_ff, d_model)) * jnp.sqrt(2.0 / d_ff)
        
        # Dynamic tanh parameters for pre-FFN normalization
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_pre_alpha'] = jax.random.uniform(subkey, (d_model,)) * 2.0 + 0.5
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_pre_beta'] = jax.random.normal(subkey, (d_model,)) * 0.1
        key, subkey = jax.random.split(key)
        params[f'ff_{layer}_pre_gamma'] = jax.random.uniform(subkey, (d_model,)) * 0.5 + 0.5
    
    return params

def init_full_transformer_params_concat(key, piece_type_vocab: int, position_vocab: int, color_vocab: int, move_vocab: int,
                                        d_model: int, n_heads: int, n_layers: int, max_seq_len: int, output_dim: int = 1) -> Dict[str, Any]:
    """Initialize complete concatenation transformer + pooling parameters"""
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    
    # Initialize transformer parameters
    transformer_params = init_transformer_params_concat(
        subkey1, piece_type_vocab, position_vocab, color_vocab, move_vocab,
        d_model, n_heads, n_layers, max_seq_len
    )
    
    # Initialize pooling parameters (same as before)
    pooling_params = init_pooling_params(subkey2, d_model, output_dim)
    
    # Combine into single parameter dictionary
    params = {**transformer_params, **pooling_params}
    
    return params

def concat_embedding_lookup(params, x):
    """
    Embedding lookup for concatenation approach with SIREN position encoding
    x shape: (seq_len, 4) where each row is [piece_type, position, color, will_move]
    Returns: (seq_len, d_model) where d_model = 4 * (d_model // 4)
    """
    seq_len, _ = x.shape
    
    # Extract component indices
    piece_types = x[:, 0].astype(jnp.int32)  # (seq_len,)
    positions = x[:, 1]  # Keep as float for SIREN differentiability
    colors = x[:, 2].astype(jnp.int32)       # (seq_len,)
    moves = x[:, 3].astype(jnp.int32)        # (seq_len,)
    
    # Lookup embeddings for categorical components
    piece_type_emb = params['piece_type_embeddings'][piece_types]  # (seq_len, d_model//4)
    color_emb = params['color_embeddings'][colors]                 # (seq_len, d_model//4)
    move_emb = params['move_embeddings'][moves]                    # (seq_len, d_model//4)
    
    # SIREN encoding for positions (with static n_layers=3)
    def encode_single_position(pos_idx):
        return siren_position_encoding(params['siren_position'], pos_idx, n_layers=3)
    
    # Vectorized SIREN encoding for all positions
    position_emb = jax.vmap(encode_single_position)(positions)  # (seq_len, d_model//4)
    
    # Concatenate all embeddings
    combined_emb = jnp.concatenate([piece_type_emb, position_emb, color_emb, move_emb], axis=-1)  # (seq_len, d_model)
    
    return combined_emb

def transformer_forward_concat(params, x, n_layers: int, n_heads: int = 8, max_seq_len: int = 32):
    """Transformer forward pass for concatenation approach"""
    # x shape: (seq_len, 4) - concatenated piece representations
    seq_len = x.shape[0]
    
    # Combined embedding lookup
    embedded = concat_embedding_lookup(params, x)  # (seq_len, d_model)
    
    # Add positional embeddings
    positions = jnp.arange(seq_len)
    pos_embedded = params['pos_embeddings'][positions]  # (seq_len, d_model)
    x_emb = embedded + pos_embedded
    
    # Create padding mask - for concatenation, we can check if piece_type is padding (6)
    piece_types = x[:, 0]
    mask = create_padding_mask_concat(piece_types)
    
    # Apply transformer layers (same as before)
    for layer_idx in range(n_layers):
        x_emb = transformer_layer(params, layer_idx, x_emb, mask, n_heads)
    
    return x_emb  # Return full sequence: (seq_len, d_model)

def create_padding_mask_concat(piece_types):
    """Create padding mask for concatenation approach based on piece types"""
    # piece_types shape: (seq_len,)
    # Padding pieces have piece_type = 6 (dead_piece_type)
    mask = (piece_types != 6).astype(jnp.float32)  # (seq_len,)
    # Create attention mask: can attend to non-padding positions
    attn_mask = jnp.outer(mask, mask)  # (seq_len, seq_len)
    return attn_mask[None, :, :]  # (1, seq_len, seq_len)

def full_transformer_forward_concat(params, x, n_layers: int, n_heads: int = 8, max_seq_len: int = 32, pooling_method: str = "mean"):
    """Complete concatenation transformer forward pass with pooling"""
    # Get sequence representations
    sequence_reps = transformer_forward_concat(params, x, n_layers, n_heads, max_seq_len)
    
    # Pool and project to output (same pooling function as before)
    output = transformer_pooling(params, sequence_reps, pooling_method)
    
    return output

def count_transformer_parameters_concat(piece_type_vocab: int, position_vocab: int, color_vocab: int, move_vocab: int,
                                        d_model: int, n_heads: int, n_layers: int, max_seq_len: int, output_dim: int = 1) -> int:
    """Count total number of parameters in the concatenation transformer with SIREN position encoding"""
    # Component embeddings (position uses SIREN network instead of embedding table)
    siren_hidden_dim = 32
    siren_n_layers = 3
    siren_params = (
        2 * siren_hidden_dim +  # First layer: input_dim=2 to hidden_dim
        (siren_n_layers - 1) * (siren_hidden_dim * siren_hidden_dim) +  # Hidden layers
        siren_hidden_dim * (d_model // 4) +  # Output layer
        (siren_n_layers + 1) * siren_hidden_dim  # All biases
    )
    
    embedding_params = (
        piece_type_vocab * (d_model // 4) +  # piece type embeddings
        siren_params +  # SIREN position network
        color_vocab * (d_model // 4) +  # color embeddings
        move_vocab * (d_model // 4) +  # move embeddings
        max_seq_len * d_model  # positional embeddings
    )
    
    # Per layer parameters (same as before)
    d_ff = int(d_model * 8 / 3)  # SwiGLU expansion ratio
    per_layer_params = (
        # Attention weights
        4 * (d_model * d_model) +  # wq, wk, wv, wo
        6 * d_model +  # pre and post dynamic tanh params (alpha, beta, gamma each)
        # SwiGLU feed-forward weights
        (d_model * d_ff) +  # w_gate
        (d_model * d_ff) +  # w_up  
        (d_ff * d_model) +  # w_down
        3 * d_model  # pre-FFN dynamic tanh params
    )
    
    # Pooling and output layer parameters (same as before)
    pooling_params = (
        3 * d_model +  # final dynamic tanh params (alpha, beta, gamma)
        (d_model * output_dim) + output_dim  # output projection w + b
    )
    
    total = embedding_params + (n_layers * per_layer_params) + pooling_params
    return total

def count_transformer_only_parameters_concat(piece_type_vocab: int, position_vocab: int, color_vocab: int, move_vocab: int,
                                             d_model: int, n_heads: int, n_layers: int, max_seq_len: int) -> int:
    """Count parameters in concatenation transformer layers only with SIREN position encoding"""
    # Component embeddings with SIREN
    siren_hidden_dim = 32
    siren_n_layers = 3
    siren_params = (
        2 * siren_hidden_dim +  # First layer
        (siren_n_layers - 1) * (siren_hidden_dim * siren_hidden_dim) +  # Hidden layers
        siren_hidden_dim * (d_model // 4) +  # Output layer
        (siren_n_layers + 1) * siren_hidden_dim  # All biases
    )
    
    embedding_params = (
        piece_type_vocab * (d_model // 4) +
        siren_params +  # SIREN instead of position embeddings
        color_vocab * (d_model // 4) +
        move_vocab * (d_model // 4) +
        max_seq_len * d_model  # positional embeddings
    )
    
    # Per layer parameters
    d_ff = int(d_model * 8 / 3)  # SwiGLU expansion ratio
    per_layer_params = (
        # Attention weights
        4 * (d_model * d_model) +  # wq, wk, wv, wo
        6 * d_model +  # pre and post dynamic tanh params (alpha, beta, gamma each)
        # SwiGLU feed-forward weights
        (d_model * d_ff) +  # w_gate
        (d_model * d_ff) +  # w_up  
        (d_ff * d_model) +  # w_down
        3 * d_model  # pre-FFN dynamic tanh params
    )
    
    total = embedding_params + (n_layers * per_layer_params)
    return total