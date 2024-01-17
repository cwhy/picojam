from __future__ import annotations

from typing import Tuple

from jax.random import normal
from jax.numpy import block, array, eye
from math import ceil, log2
from jax.typing import ArrayLike
from random_utils import SafeKey

def normal_init(rng_key: SafeKey, sd: float, shape: Tuple[int, ...]) -> ArrayLike:
    return normal(rng_key.get(), shape) * sd


def hadamard_matrix(size: int) -> ArrayLike:
    """ Generate a Hadamard matrix of given size """
    # Check if size is a power of 2, as Hadamard matrices are only defined for these sizes
    assert (size & (size - 1) == 0) and size > 0, "Size must be a power of 2"

    # Base case
    if size == 1:
        return array([[1]])

    # Recursive construction
    h = hadamard_matrix(size // 2)
    return block([[h, h], [h, -h]])

def zerO_init_2D(shape: Tuple[int, int]) -> ArrayLike:
    m, n = shape
    if m <= n:
        return eye(m, n)
    else:
        clog_m = ceil(log2(m))
        p = 2 ** clog_m
        return eye(m, p) @ (hadamard_matrix(p) / (2 ** (clog_m / 2))) @ eye(p, n)
