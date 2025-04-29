from jax import Array
import jax.numpy as jnp


import os, sys

def create_coordinate_indices(height: int, width: int) -> Array:
    y_indices = jnp.arange(height)
    x_indices = jnp.arange(width)
    
    y_grid, x_grid = jnp.meshgrid(y_indices, x_indices, indexing='ij')
    coords = jnp.stack([y_grid.flatten(), x_grid.flatten()], axis=1)
    
    coords = coords.astype(jnp.float32)
    coords = coords / jnp.array([height - 1, width - 1])
    
    return coords

