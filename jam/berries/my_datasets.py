from datasets import load_dataset
from typing import NamedTuple
from jax import Array


class Supervised1D(NamedTuple):
    n_samples: int
    d_x: int
    d_y: int
    X: Array
    y: Array
    X_test: Array
    y_test: Array

class ImageClassification(NamedTuple):
    n_samples: int
    d_x: tuple[int, int]
    d_y: int
    n_channels: int
    X: Array
    y: Array
    X_test: Array
    y_test: Array
    

cache_dir="$HOME/.cache/huggingface/datasets"

def load_supervised_1d(data: str) -> Supervised1D:
    if data == "mnist":
        mnist = load_dataset("mnist", cache_dir=cache_dir).with_format("jax")
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
        return Supervised1D(n_samples, d_x, d_y, X, y, X_test, y_test)
    elif data == "fashion_mnist":
        fashion_mnist = load_dataset("fashion_mnist", cache_dir=cache_dir).with_format("jax")
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
        return Supervised1D(n_samples, d_x, d_y, X, y, X_test, y_test)

def load_supervised_image(data: str) -> ImageClassification:
    if data == "mnist":
        mnist = load_dataset("mnist", cache_dir=cache_dir, trust_remote_code=True).with_format("jax")
        mnistData = mnist['train']
        X_img = mnistData['image']
        y = mnistData['label']
        X_img_test = mnist["test"]["image"]
        n_test_samples = X_img_test.shape[0]
        y_test = mnist["test"]["label"]
        n_samples, _, _  = X_img.shape
        X_train = X_img.reshape((n_samples, 1, 28, 28))
        X_test = X_img_test.reshape((n_test_samples, 1, 28, 28))
        n_channels = 1
        d_x = (28, 28)
        d_y = len(set(y.tolist()))
        return ImageClassification(n_samples, d_x, d_y, n_channels, X_train, y, X_test, y_test)
    elif data == "fashion_mnist":
        fashion_mnist = load_dataset("fashion_mnist", cache_dir=cache_dir).with_format("jax")
        fashion_mnistData = fashion_mnist['train']
        X_img = fashion_mnistData['image']
        y = fashion_mnistData['label']
        X_img_test = fashion_mnist["test"]["image"]
        n_test_samples = X_img_test.shape[0]
        y_test = fashion_mnist["test"]["label"]
        n_samples, _, _  = X_img.shape
        X_train = X_img.reshape((n_samples, 1, 28, 28))
        X_test = X_img_test.reshape((n_test_samples, 1, 28, 28))
        n_channels = 1
        d_x = (28, 28)
        d_y = len(set(y.tolist()))
        return ImageClassification(n_samples, d_x, d_y, n_channels, X_train, y, X_test, y_test)
    elif data == "cifar10":
        cifar10 = load_dataset("uoft-cs/cifar10", cache_dir=cache_dir).with_format("jax")
        cifar10Data = cifar10['train']
        X_img = cifar10Data['image']
        y = cifar10Data['label']
        X_img_test = cifar10["test"]["image"]
        n_test_samples = X_img_test.shape[0]
        y_test = cifar10["test"]["label"]
        n_samples, _, _, _ = X_img.shape
        X_train = X_img.transpose((0, 3, 1, 2))
        X_test = X_img_test.transpose((0, 3, 1, 2))
        n_channels = 3
        d_x = (32, 32)
        d_y = len(set(y.tolist()))
        return ImageClassification(n_samples, d_x, d_y, n_channels, X_train, y, X_test, y_test)
    