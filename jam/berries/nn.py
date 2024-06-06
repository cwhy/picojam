from jax.nn import sigmoid
from jax.numpy.linalg import norm
from jax.numpy import sqrt
from pf import F, _

def affine(x, W, b):
    return W.T @ x + b

def swish(x):
    return x * sigmoid(x)

def sglu(x, wv, wu, wo):
    v = x @ wv
    u = x @ wu
    return (v * u) @ wo

def rmsn(x, d):
    return  x / (norm(x)/ sqrt(d))

def mglu(x, ws):
    for w in ws:
        sglu_w = F(sglu).f(_, **w['sglu'])
        x = rmsn(sglu_w(x), **w['rmsn'])
    return x


def mglu_net(x, ws):
    mglu_ws, sglu_w = ws['mglu'], ws['sglu']
    x = mglu(x, mglu_ws)
    return sglu(x, **sglu_w)

def rmglu(x, ws):
    for w in ws:
        sglu_w = F(sglu).f(_, **w['sglu'])
        x += rmsn(sglu_w(x), **w['rmsn'])
    return x

def rmglu_net(x, ws):
    mglu_ws, sglu_w = ws['mglu'], ws['sglu']
    x = rmglu(x, mglu_ws)
    return sglu(x, **sglu_w)

import init_utils, random_utils
from init_utils import zerO_init_2D
from jax.random import split

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

def rmsn_config(d_out):
    return {
        "d": {
            "const": float(d_out),
        }
    }

def mglu_layer_config(d_in, d_h, d_out, init):
    return {
        "sglu": sglu_config(d_in, d_h, d_out, init),
        "rmsn": rmsn_config(d_out),
    }


def mglu_config(d_in, d_h_layer, d_out, d_h, n_layers, init):
    return tuple([
        mglu_layer_config(d_in, d_h, d_h_layer, init),
        *[mglu_layer_config(d_h_layer, d_h, d_h_layer, init)] * (n_layers - 1),
        mglu_layer_config(d_h_layer, d_h, d_out, init),
    ])


def mglu_net_config(d_in, d_h_layer, d_out, d_h, n_layers, init):
    return {
        "mglu": mglu_config(d_in, d_h_layer, d_h_layer, d_h, n_layers - 1, init),
        "sglu": sglu_config(d_h_layer, d_h, d_out, init),
    }



def init_weight(key, init, size):
    if init["type"] == "normal":
        return init_utils.normal_init(key, init["std"], size)
    elif init["type"] == "zer0":
        return zerO_init_2D(size)
    else:
        raise ValueError(f"Unknown init type: {init['type']}")


def init_weights(key, configs):
    if isinstance(configs, tuple):
        keys = key.split(len(configs))
        return tuple(init_weights(k, c) for c, k in zip(configs, keys))
    elif isinstance(configs, dict):
        if "const" in configs:
            return configs["const"]
        if "init" in configs:
            return init_weight(key, **configs)
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