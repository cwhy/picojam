from jax.numpy import array, sqrt, meshgrid, arange
from nnn import gaussian_activation, rmsnorm, sglu
from pf import F, _

d_x = (28, 28)
d_y = 10
pos_x1, pos_x2 = meshgrid(arange(d_x[0]), arange(d_x[1]))
pos_x1 = pos_x1.flatten()
pos_x2 = pos_x2.flatten()

d_mglu_h_layer = 32
d_mglu_h = 32
n_mglu_layers = 2

n_mixer_layers = 2
d_mixer_channels = 16

d_encode_hidden = 32
h_axis = 1
x_axis = 0
d_h_axis = d_mixer_channels
d_x_axis = d_x[0] * d_x[1]

def pos_encode_v(W, x1, x2, v):
    rep = array([x1, x2, v])
    l1 = W['0'] @ rep
    a = W['a']
    a1 = gaussian_activation(a, l1)
    return W['1'] @ a1

def pos_encode(W, x1, x2):
    rep = array([x1, x2])
    l1 = W['0'] @ rep
    a = W['a']
    a1 = gaussian_activation(a, l1)
    return W['1'] @ a1

def mixer_head(W, x):
    val_flat = x.reshape(-1)
    return F(pos_encode_v).vmap((None, 0, 0, 0), 0)(W, pos_x1, pos_x2, val_flat)

def mix_x(W):
    w_in = F(pos_encode).vmap((None, 0, 0), 0)(W['in'], pos_x1, pos_x2)
    w_out = F(pos_encode).vmap((None, 0, 0), 0)(W['out'], pos_x1, pos_x2)
    return (w_out @ w_in.T) / sqrt(d_mixer_channels)


def pure_mixer(W, x):
    X = mixer_head(W['head'], x)
    w_x = mix_x(W['mix_x'])
    mixer_h = F(sglu).f(_, **W['h']).vmap(x_axis, x_axis)
    mixer_x = F(lambda x: w_x @ x).vmap(h_axis, h_axis)
    norm_h = F(rmsnorm).vmap(None, h_axis)
    norm_x = F(rmsnorm).vmap(None, x_axis)
    for layer in range(n_mixer_layers):
        X_x = norm_x(W['rmsnorm_x'][:, layer], mixer_x(X))
        X += X_x
        X_h = norm_h(W['rmsnorm_h'][:, layer], mixer_h(X))
        X += X_h
    return X


def pure_mixer_net(W, x):
    X = pure_mixer(W, x) 
    return sglu(X.sum(axis=x_axis), **W['out'])


def pos_config(d_pos, init):
    return {
        'a': {
            "size": (d_encode_hidden,),
            "const": 1.0
        },
        '0': {
            "size": ( d_encode_hidden, d_pos),
            "init": {
                "std": init['std'],
                "type": "normal"
            }
        },
        '1': {
            "size": (d_mixer_channels, d_encode_hidden),
            "init": {
                "std": init['std'],
                "type": "normal"
            }
        }
    }
mixer_head_config = F(pos_config).f(3, _)

def mix_x_config(init):
    return {
        'in': pos_config(2, init),
        'out': pos_config(2, init)
    }

def pure_mixer_config(init):
    return {
        'head': mixer_head_config(init),
        'mix_x': mix_x_config(init),
        'h': mglu_net_conf_local(d_h_axis, d_h_axis, init),
        'out': mglu_net_conf_local(d_mixer_channels, d_y, init),
        'rmsnorm_x': {
            'size': (d_mixer_channels, n_mixer_layers),
            'const': 1
        },
        'rmsnorm_h': {
            'size': (d_mixer_channels, n_mixer_layers),
            'const': 1
        }

    }


def mixer_config(init):
    return {
        'mixer': {
            'head': mixer_head_config(init),
            'block': mixer_block_config(init),
        },
        'out': mglu_net_conf_local(d_mixer_channels, d_y, init)
            #sglu_config(d_mixer_channels, d_mglu_h,  d_y, init)
    }