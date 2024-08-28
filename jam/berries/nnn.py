
from jax.nn import sigmoid
from jax.numpy.linalg import norm
from jax.numpy import ones, mean, square, exp, sqrt
from jax.lax import rsqrt
from init_utils import zerO_init_2D, normal_init
from pf import F, _


EPS = 1e-8

def flatten(x):
    return x.reshape(-1)

def affine(x, W, b):
    return W.T @ x + b

def rmsnorm(w, x, eps=EPS):
    x_norm = x * rsqrt(mean(x ** 2) + eps)
    return x_norm * w


def swish(x):
    return x * sigmoid(x)

def sglu(x, wv, wu, wo):
    v = x @ wv
    u = x @ wu
    return (v * u) @ wo

def gaussian_activation(a, x):
    return exp((-0.5 * x ** 2) / a ** 2)



def W_config(d_in, d_out, init):
    return {
        "size": (d_in, d_out),
        "init": init,
    }

        

def sglu_config(d_in, d_h, d_out, init):
    return {
        "wv": {
            "size": (d_in, d_h),
            "init": init,
        },
        "wu": {
            "size": (d_in, d_h),
            "init": init,
        },
        "wo": {
            "size": (d_h, d_out),
            "init": init,
        },
    }

def init_weight(key, configs):
    if "const" in configs:
        if "size" in configs:
            return configs["const"] * ones(configs["size"])
        else:
            return configs["const"]
    if "init" in configs:
        init, size = configs["init"], configs["size"]
        if init["type"] == "xavier":
            return normal_init(key, sqrt(2 / (size[0] + size[1])), size)
        elif init["type"] == "normal":
            return normal_init(key, init["std"], size)
        elif init["type"] == "zer0":
            return zerO_init_2D(size)
        else:
            raise ValueError(f"Unknown init type: {init['type']}")


def init_weights(key, configs):
    if isinstance(configs, tuple):
        keys = key.split(len(configs))
        return tuple(init_weights(k, c) for c, k in zip(configs, keys))
    elif isinstance(configs, dict):
        if "const" in configs or "init" in configs:
            return init_weight(key, configs)
        else:
            keys = key.split(len(configs))
            return {name: init_weights(k, config) for (name, config), k in zip(configs.items(), keys)}


def fmt_weights(weights, indent=0):
    out_str = ""
    out_n_params = 0
    if isinstance(weights, tuple):
        for w in weights:
            sub_out_str, sub_n_params = fmt_weights(w, indent + 4)
            out_str += f"{' ' * indent}tuple:\n{sub_out_str}"
            out_n_params += sub_n_params
        out_str += f"{' ' * indent}total params: {out_n_params}\n"
    elif isinstance(weights, dict):
        for name, w in weights.items():
            sub_out_str, sub_n_params = fmt_weights(w, indent + 4)
            out_str += f"{' ' * indent}{name}:\n{sub_out_str}"
            out_n_params += sub_n_params
        out_str += f"{' ' * indent}total params: {out_n_params}\n"
    else:
        if hasattr(weights, 'shape'):
            out_n_params += weights.size
            out_str += f"{' ' * indent}array shape: {weights.shape}\n"
        else:
            out_n_params += 1
            out_str += f"{' ' * indent}{weights}\n"
    return out_str, out_n_params