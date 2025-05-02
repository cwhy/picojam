from functools import cache
from jax import Array, device_get
from jax.numpy import arange, stack, array, meshgrid, float32, maximum
import jax.numpy as jnp
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap



@cache
def create_coordinate_indices(height: int, width: int) -> Array:
    y_indices = arange(height)
    x_indices = arange(width)
    
    y_grid, x_grid = meshgrid(y_indices, x_indices, indexing='ij')
    coords = stack([y_grid.flatten(), x_grid.flatten()], axis=1)
    return coords

# coords = coords.astype(float32)
# coords = coords / array([height - 1, width - 1])

@cache
def make_positional_color_map(img_dims: tuple[int, int]) -> jnp.ndarray:
    """
    Returns an array of shape (H*W, 3) with colors from a perceptual colormap.
    Encodes spatial position into a scalar, then maps that to RGB using viridis.
    """
    h, w = img_dims
    coords = create_coordinate_indices(h, w)  # shape (H*W, 2)

    # Normalize to range [0, 1]
    ys = coords[:, 0] / jnp.maximum(h - 1, 1)
    xs = coords[:, 1] / jnp.maximum(w - 1, 1)

    # Example scalar field: angle from center (could also use distance or other functions)
    cx, cy = 0.5, 0.5
    dx = xs - cx
    dy = ys - cy
    angle = jnp.arctan2(dy, dx)  # range [-π, π]
    angle_norm = (angle + jnp.pi) / (2 * jnp.pi)  # normalize to [0,1]

    # Map to RGB using viridis colormap (in host memory)
    viridis = get_cmap("viridis")
    colors = viridis(device_get(angle_norm))[:, :3]  # (H*W, 3), drop alpha

    return jnp.array(colors, dtype=jnp.float32)

@cache
def make_positional_colormap_for_matplotlib(img_dims):
    arr = make_positional_color_map(img_dims)
    return ListedColormap(device_get(arr)) 