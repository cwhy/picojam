import argparse
import logging
import os
import sys
from functools import partial
from typing import Any, Dict, Iterator, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from optimizer_bank import get_optimizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from berries.my_datasets import ImageClassification, load_supervised_image
from berries.random_utils import SafeKey, infinite_safe_keys_from_key


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

IMG_H = 28
IMG_W = 28
SEQ_LEN = IMG_H * IMG_W
NUM_CLASSES = 10
NUM_PIXEL_VALUES = 256


def default_config() -> Dict[str, Any]:
    return {
        "dataset_name": "mnist",
        "num_epochs": 10,
        "num_train_samples": 10_000,
        "num_test_samples": 2_000,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "optimizer": "adamw",
        "random_seed": 42,
        "d_model": 64,
        "hidden_dim": 128,
        "recurrent_scale": 0.85,
        "eval_every": 1,
    }


def init_params(config: Dict[str, Any]) -> Tuple[Dict[str, Array], Iterator[SafeKey]]:
    key = jax.random.PRNGKey(config["random_seed"])
    key_gen = infinite_safe_keys_from_key(key)

    d_model = config["d_model"]
    hidden_dim = config["hidden_dim"]
    recurrent_scale = config["recurrent_scale"]

    params = {
        "position_embedding": jax.random.normal(next(key_gen).get(), (SEQ_LEN, d_model)) * 0.02,
        "value_embedding": jax.random.normal(next(key_gen).get(), (NUM_PIXEL_VALUES, d_model)) * 0.02,
        "A": jax.random.normal(next(key_gen).get(), (hidden_dim, hidden_dim))
        * (recurrent_scale / jnp.sqrt(hidden_dim)),
        "B": jax.random.normal(next(key_gen).get(), (hidden_dim, d_model)) * (1.0 / jnp.sqrt(d_model)),
        "b": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "W_out": jax.random.normal(next(key_gen).get(), (NUM_CLASSES, hidden_dim)) * (1.0 / jnp.sqrt(hidden_dim)),
        "c": jnp.zeros((NUM_CLASSES,), dtype=jnp.float32),
    }
    return params, key_gen


def _forward_single(params: Dict[str, Array], pixel_tokens: Array) -> Array:
    x_seq = params["position_embedding"] + params["value_embedding"][pixel_tokens]

    def step(h_prev: Array, x_t: Array) -> Tuple[Array, None]:
        h_next = params["A"] @ h_prev + params["B"] @ x_t + params["b"]
        return h_next, None

    hidden_dim = params["A"].shape[0]
    h0 = jnp.zeros((hidden_dim,), dtype=jnp.float32)
    h_T, _ = jax.lax.scan(step, h0, x_seq)
    logits = params["W_out"] @ h_T + params["c"]
    return logits


def forward_batch(params: Dict[str, Array], batch_pixel_tokens: Array) -> Array:
    return jax.vmap(_forward_single, in_axes=(None, 0))(params, batch_pixel_tokens)


def loss_batch(params: Dict[str, Array], batch_pixel_tokens: Array, batch_labels: Array) -> Array:
    logits = forward_batch(params, batch_pixel_tokens)
    loss_per_example = optax.softmax_cross_entropy_with_integer_labels(logits, batch_labels)
    return jnp.mean(loss_per_example)


@partial(jax.jit, static_argnums=(0,))
def train_step(
    optimizer: optax.GradientTransformation,
    params: Dict[str, Array],
    opt_state: optax.OptState,
    batch_pixel_tokens: Array,
    batch_labels: Array,
) -> Tuple[Dict[str, Array], optax.OptState, Array]:
    loss_value, grads = jax.value_and_grad(loss_batch)(params, batch_pixel_tokens, batch_labels)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value


@jax.jit
def predict_batch(params: Dict[str, Array], batch_pixel_tokens: Array) -> Array:
    logits = forward_batch(params, batch_pixel_tokens)
    return jnp.argmax(logits, axis=-1)


def evaluate_accuracy(params: Dict[str, Array], X: Array, y: Array, batch_size: int) -> float:
    total = X.shape[0]
    total_correct = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        pred = predict_batch(params, X[start:end])
        total_correct += int(jnp.sum(pred == y[start:end]))
    return total_correct / total


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "num_epochs": args.epochs,
        "num_train_samples": args.train_samples,
        "num_test_samples": args.test_samples,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "d_model": args.d_model,
        "hidden_dim": args.hidden_dim,
        "eval_every": args.eval_every,
        "random_seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear single-layer RNN MNIST demo with learnable position/value embeddings."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_config()
    config = apply_overrides(config, args)
    params, key_gen = init_params(config)

    if config["batch_size"] > config["num_train_samples"]:
        raise ValueError("batch_size cannot exceed num_train_samples.")

    logging.info("Loading MNIST dataset...")
    data: ImageClassification = load_supervised_image(
        config["dataset_name"],
        n_tr=config["num_train_samples"],
        n_tst=config["num_test_samples"],
    )

    X_train = data.X.reshape(data.n_samples, -1).astype(jnp.int32)
    y_train = data.y.astype(jnp.int32)
    X_test = data.X_test.reshape(data.n_test_samples, -1).astype(jnp.int32)
    y_test = data.y_test.astype(jnp.int32)

    optimizer = get_optimizer(config)
    opt_state = optimizer.init(params)

    n_train = X_train.shape[0]
    batch_size = config["batch_size"]
    num_batches = n_train // batch_size

    logging.info("Starting training with config: %s", config)
    for epoch in range(1, config["num_epochs"] + 1):
        permutation = jax.random.permutation(next(key_gen).get(), n_train)
        x_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            params, opt_state, loss_value = train_step(
                optimizer,
                params,
                opt_state,
                x_shuffled[start:end],
                y_shuffled[start:end],
            )
            epoch_loss += float(loss_value)

        avg_loss = epoch_loss / num_batches
        if epoch % config["eval_every"] == 0 or epoch == 1 or epoch == config["num_epochs"]:
            acc = evaluate_accuracy(params, X_test, y_test, batch_size=batch_size)
            logging.info(
                "Epoch %d/%d | train_loss=%.4f | test_acc=%.4f",
                epoch,
                config["num_epochs"],
                avg_loss,
                acc,
            )
        else:
            logging.info("Epoch %d/%d | train_loss=%.4f", epoch, config["num_epochs"], avg_loss)


if __name__ == "__main__":
    main()
