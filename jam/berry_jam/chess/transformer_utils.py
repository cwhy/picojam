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