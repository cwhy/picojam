from datasets import load_dataset
from typing import NamedTuple
from jax import Array


class Supervised(NamedTuple):
    n_samples: int
    d_x: int
    d_y: int
    X: Array
    y: Array
    X_test: Array
    y_test: Array
    


def load_supervised(data: str) -> Supervised:
    if data == "mnist":
        mnist = load_dataset("mnist").with_format("jax")
        mnistData = mnist['train']
        X_img = mnistData['image']
        y = mnistData['label']
        X_img_test = mnist["test"]["image"]
        n_test_samples = X_img_test.shape[0]
        X_test = X_img_test.reshape((n_test_samples, -1))
        y_test = mnist["test"]["label"]
        n_samples, _, _ = X_img.shape
        X = X_img.reshape((n_samples, -1))
        n_samples, d_x = X.shape
        d_y = len(set(y.tolist()))
        return Supervised(n_samples, d_x, d_y, X, y, X_test, y_test)
    elif data == "fashion_mnist":
        fashion_mnist = load_dataset("fashion_mnist").with_format("jax")
        fashion_mnistData = fashion_mnist['train']
        X_img = fashion_mnistData['image']
        y = fashion_mnistData['label']
        X_img_test = fashion_mnist["test"]["image"]
        n_test_samples = X_img_test.shape[0]
        X_test = X_img_test.reshape((n_test_samples, -1))
        y_test = fashion_mnist["test"]["label"]
        n_samples, _, _ = X_img.shape
        X = X_img.reshape((n_samples, -1))
        n_samples, d_x = X.shape
        d_y = len(set(y.tolist()))
        return Supervised(n_samples, d_x, d_y, X, y, X_test, y_test)
