# Brute forcing a torch state_dict to flax params implementation

import torch
import flax

from jax import numpy as jnp

from src.util.jax_utils import get_or_make


def _is_embedding(name):
    return '.wte.' in name or '.wpe.' in name


def _is_layer_norm(name):
    return '.ln_' in name


def _is_attention(name):
    return '.attn.' in name


def _is_mlp(name):
    return '.mlp.' in name


def map_torch_name_to_flax_key(name):

    path = name.split('.')
    if '_w_u' in path:  # Sketchy, can we do better?
        # it's a rotation matrix
        idx_l = [name[-1]]
        if len(path > 4):
            if _is_layer_norm(name):
                return path[:-1] + ['scale'] + idx_l, True
            elif _is_attention(name) or _is_mlp(name):
                return path[:-1] + ['kernel'] + idx_l, True
        else:
            if _is_embedding(name):
                return path[:-2] + ['embedding'] + idx_l, True
            elif _is_layer_norm(name):
                return path[:-2] + ['scale'] + idx_l, True
            else:
                return None, True  # it's related the lm_head
    else:
        # it's a real parameter
        if len(path) > 3:
            if _is_layer_norm(name):
                return path[:-1] + ['scale'], False
            elif _is_attention(name) or _is_mlp(name):
                return path[:-1] + ['kernel'], False
        else:
            if _is_embedding(name):
                return path[:-1] + ['embedding'], False
            elif _is_layer_norm(name):
                return path[:-1] + ['scale'], False
            else:
                return None, False  # it's the lm_head

    raise AttributeError(f'{name} is not recognized as a type of weight in torch')


def transfer_state_dict_to_params(state_dict, params):
    for name, param in state_dict.items():
        path, is_a_rotation_matrix = map_torch_name_to_flax_key(name)
        if path is None:
            continue
        d = params
        while len(path) > 1:
            d = d[path.pop(0)]
        if path[0] in d:
            d[path[0]] = jnp.asarray(param.detach().cpu().numpy())
        else:
            print(f'{path} doesn\'t exist! skipping')
    return params


def transfer_state_dict_to_params_and_rotations(state_dict, params):
    rotations = {}
    for name, param in state_dict.items():
        path, is_a_rotation_matrix = map_torch_name_to_flax_key(name)
        if path is None:
            continue
        d = rotations if is_a_rotation_matrix else params
        while len(path) > (2 if is_a_rotation_matrix else 1):
            d = get_or_make(d, path.pop(0))
        w = jnp.asarray(param.detach().cpu().numpy())
        if is_a_rotation_matrix:
            key = path.pop(0)
            idx = path[0]
            if key not in d:
                d[key] = [w]
            else:
                d[key].insert(idx, w)
        else:
            if path[0] in d:
                d[path[0]] = w
            else:
                print(f'{path} doesn\'t exist! skipping')
    return params, rotations
