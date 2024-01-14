from jax.numpy import pi, power, sin, arange
from jax import vmap
from jax.typing import ArrayLike

# pos only take scalar value
def positional_encoding(pos: ArrayLike, dim: int) -> ArrayLike:
    hidden_seq = arange(dim)
    # Calculate angle rates
    angle_rads = pos / power(10000, (2 * (hidden_seq // 2) / float(dim)))

    # Create an array with 0 for even indices and pi/2 for odd indices
    phase_shift = pi / 2 * (hidden_seq % 2)

    # Apply the phase shift
    angle_rads_shifted = angle_rads + phase_shift

    return sin(angle_rads_shifted)

def get_positional_encoding(max_len: int, dim: int) -> ArrayLike:
    positional_encoding_v = vmap(positional_encoding, in_axes=(0, None))
    return positional_encoding_v(arange(max_len), dim)