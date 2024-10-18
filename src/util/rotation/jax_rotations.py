import copy
from abc import abstractmethod
from itertools import product

import jax
import numpy as np
from jax import numpy as jnp
from jax.numpy import linalg as jla
import jax.tree_util as tu
import jax.flatten_util as fu

from src.util.torch_to_flax import _is_mlp, _is_embedding, _is_attention, _is_layer_norm
from src.util.jax_utils import tree_path_to_name, random_split_like_tree

WEIGHT_MAP = {
    'mlp': _is_mlp,
    'attention': _is_attention,
    'layernorm': _is_layer_norm,
    'embedding': _is_embedding
}

FACTORIZATION_MAP = {
    'attention': [[512, 384, 108], [256, 384, 72]],
    'mlp': [[512, 384, 144], [512, 384, 144]],
    'embedding': [[384, 401, 256]],
    'layernorm': [[128, 150]]
}


class BaseRotation:

    @staticmethod
    def from_checkpoint(checkpoint, params, rotations, **kwargs): raise NotImplementedError()

    @staticmethod
    def build(params, rng, **kwargs): raise NotImplementedError()


def recover_rotations(rotation_builders, checkpoint_state, params, rotations):
    forward_fns = []
    backward_fns = []
    for i in range(len(rotation_builders)):
        state, forward_fn, backward_fn = rotation_builders[i](checkpoint_state[i], params, rotations)
        forward_fns.append(forward_fn)
        backward_fns.append(backward_fn)
    return forward_fns, backward_fns


def make_rotations(params, rotation_builders, rng):
    rngs = jax.random.split(rng, len(rotation_builders))
    forward_fns = []
    backward_fns = []
    for i in range(len(rotation_builders)):
        state, forward_fn, backward_fn = rotation_builders[i](params, rngs[i])
        forward_fns.append(forward_fn)
        backward_fns.append(backward_fn)
    return forward_fns, backward_fns


class RandomPermuteRotation(BaseRotation):
    factorizations = [[883, 139, 725], [256, 256, 540]]

    @staticmethod
    def from_checkpoint(checkpoint_state, params, rotations):
        RPR = RandomPermuteRotation
        torch_perms, torch_rotators = checkpoint_state[0], checkpoint_state[1]
        perms, rotators = [], []
        k = len(torch_perms)
        for i in range(k):
            perms.append([])
            rotators.append([])
            for j in range(2):
                perms[-1].append(jnp.asarray(torch_perms[i][j].detach().cpu().numpy()))
                rotation_matrices = []
                for rotator in torch_rotators[i][j]:
                    rotation_matrices.append(jnp.asarray(rotator.detach().cpu().numpy()))
                rotators[-1].append(rotation_matrices)
        inverse_perms = [[jnp.argsort(perms[i][j]) for j in range(2)] for i in range(k)]
        state = [rotators, perms, inverse_perms]
        forward_fn = RPR._make_forward_fn(state, k)
        backward_fn = RPR._make_backward_fn(state, k)
        return state, forward_fn, backward_fn

    @staticmethod
    def build(params, rng, k):
        RPR = RandomPermuteRotation
        vec_1, vec_2 = _make_vecs(params)
        ps = vec_1.shape[0], vec_2.shape[0]
        rngs = jax.random.split(rng, 8 * k)
        perms = [[jax.random.permutation(rngs[i * k + j], ps[j]) for j in range(2)] for i in range(k)]
        rngs = rngs[2*k:]
        rotators = [[
            [_gaussian_orthogonal_matrix(d, d, rngs[i * k + j * 2 + a]) for a, d in enumerate(RPR.factorizations[j])] for j in range(2)
        ] for i in range(k)]
        inverse_perms = [[jnp.argsort(perms[i][j]) for j in range(2)] for i in range(k)]
        state = [rotators, perms, inverse_perms]
        forward_fn = RPR._make_forward_fn(state, k)
        backward_fn = RPR._make_backward_fn(state, k)
        return state, forward_fn, backward_fn

    @staticmethod
    def _make_forward_fn(state, k):
        def forward(params):
            vecs = list(_make_vecs(params))
            for i in range(k):
                for j in range(2):
                    perm = state[1][i][j]
                    orthogonal_rotators = state[0][i][j]
                    vecs[j] = _apply_rotators(orthogonal_rotators, vecs[j][perm])
            return _reconstruct_params(vecs, params)
        return forward

    @staticmethod
    def _make_backward_fn(state, k):
        def backward(params):
            vecs = list(_make_vecs(params))
            for i in reversed(range(k)):
                for j in range(2):
                    perm = state[2][i][j]
                    orthogonal_rotators = state[0][i][j]
                    vecs[j] = _apply_rotators(orthogonal_rotators, vecs[j], transpose=True)[perm]
            return _reconstruct_params(vecs, params)
        return backward


class WeightRandPermRotation(BaseRotation):

    @staticmethod
    def from_checkpoint(checkpoint_state, params, rotations, weight_types):
        WRP = WeightRandPermRotation
        filter_fn = WEIGHT_MAP[weight_types[0]]
        factorizations = FACTORIZATION_MAP[weight_types[0]]  # no need
        torch_perms, torch_rotators = checkpoint_state[0], checkpoint_state[1]
        perms, rotators = [], []
        k = len(torch_perms)
        for i in range(k):
            perms.append([])
            rotators.append([])
            for j in range(len(torch_perms[i])):
                perms[-1].append(jnp.asarray(torch_perms[i][j].detach().cpu().numpy()))
                rotation_matrices = []
                for rotator in torch_rotators[i][j]:
                    rotation_matrices.append(jnp.asarray(rotator.detach().cpu().numpy()))
                rotators[-1].append(rotation_matrices)
        inverse_perms = [[np.argsort(perms[i][j]) for j in range(len(factorizations))] for i in range(k)]
        state = [rotators, perms, inverse_perms]
        forward_fn = WRP._make_forward_fn(state, k, filter_fn)
        backward_fn = WRP._make_backward_fn(state, k, filter_fn)
        return state, forward_fn, backward_fn

    @staticmethod
    def build(params, rng, k, weight_types):
        WRP = WeightRandPermRotation
        filter_fn = WEIGHT_MAP[weight_types[0]]
        factorizations = FACTORIZATION_MAP[weight_types[0]]  # no need
        vecs_inc = _make_vecs_with_filter(params, filter_fn)[0]
        ps = [v.shape[0] for v in vecs_inc]
        rngs = jax.random.split(rng, 8 * k)
        perms = [[jax.random.permutation(rngs[i * k + j], ps[j]) for j in range(len(ps))] for i in range(k)]
        rngs = rngs[2*k:]
        rotators = [[
            [_gaussian_orthogonal_matrix(d, d, rngs[i * k + j * 2 + a]) for a, d in enumerate(factorizations[j])] for j in range(len(ps))
        ] for i in range(k)]
        inverse_perms = [[jnp.argsort(perms[i][j]) for j in range(len(ps))] for i in range(k)]
        state = [rotators, perms, inverse_perms]
        forward_fn = WRP._make_forward_fn(state, k, filter_fn)
        backward_fn = WRP._make_backward_fn(state, k, filter_fn)
        return state, forward_fn, backward_fn

    @staticmethod
    def _make_forward_fn(state, k, filter_fn):
        def forward(params):
            vecs_inc = list(_make_vecs_with_filter(params, filter_fn)[0])
            for i in range(k):
                for j in range(len(vecs_inc)):
                    perm = state[1][i][j]
                    orthogonal_rotators = state[0][i][j]
                    vecs_inc[j] = _apply_rotators(orthogonal_rotators, vecs_inc[j][perm])
            return _reconstruct_params_with_filter(vecs_inc, params, filter_fn)
        return forward

    @staticmethod
    def _make_backward_fn(state, k, filter_fn):
        def backward(params):
            vecs_inc = list(_make_vecs_with_filter(params, filter_fn)[0])
            for i in reversed(range(k)):
                for j in range(len(vecs_inc)):
                    perm = state[2][i][j]
                    orthogonal_rotators = state[0][i][j]
                    vecs_inc[j] = _apply_rotators(orthogonal_rotators, vecs_inc[j], transpose=True)[perm]
            return _reconstruct_params_with_filter(vecs_inc, params, filter_fn)
        return backward


class WeightRotation(BaseRotation):

    @staticmethod
    def from_checkpoint(checkpoint_state, params, rotations):
        state = rotations
        forward_fn = WeightRotation._make_forward_fn(state)
        backward_fn = WeightRotation._make_backward_fn(state)
        return state, forward_fn, backward_fn

    @staticmethod
    def build(params, rng, weight_types):
        functions = list(map(lambda weight: WEIGHT_MAP[weight], weight_types))
        rngs = random_split_like_tree(rng, params)
        state = _make_orthogonal_buffers(params, rngs, functions)
        forward_fn = WeightRotation._make_forward_fn(state)
        backward_fn = WeightRotation._make_backward_fn(state)
        return state, forward_fn, backward_fn

    @staticmethod
    def _make_forward_fn(states):
        def forward(param, state):
            if state is not None:
                if len(state) == 1:
                    return state[0] @ param
                elif len(state) == 2:
                    return state[0] @ param @ state[1]
            return param
        # Todo: fix this, this is broken because jax drops nones instead of treating them as leaves!
        return lambda params: jax.tree_util.tree_map(forward, params, states, is_leaf=lambda x: x is None)

    @staticmethod
    def _make_backward_fn(states):
        def backward(param, state):
            if state is not None:
                if len(state) == 1:
                    return state[0].T @ param
                else:
                    return state[0].T @ param @ state[1].T
            return param
        return lambda params: jax.tree_util.tree_map(backward, params, states, is_leaf=lambda x: x is None)


class BlockMixingRotation(BaseRotation):

    _mappers = [
        lambda block: block['ln_1']['scale'],
        lambda block: block['attn']['c_attn']['kernel'],
        lambda block: block['attn']['c_proj']['kernel'],
        lambda block: block['ln_2']['scale'],
        lambda block: block['mlp']['c_fc']['kernel'],
        lambda block: block['mlp']['c_proj']['kernel'],
    ]
    _setters = [
        lambda block, value: _setter(block, ['ln_1', 'scale'], value),
        lambda block, value: _setter(block, ['attn', 'c_attn', 'kernel'], value),
        lambda block, value: _setter(block, ['attn', 'c_proj', 'kernel'], value),
        lambda block, value: _setter(block, ['ln_2', 'scale'], value),
        lambda block, value: _setter(block, ['mlp', 'c_fc', 'kernel'], value),
        lambda block, value: _setter(block, ['mlp', 'c_proj', 'kernel'], value),
    ]

    @staticmethod
    def from_checkpoint(checkpoint_state, params, rotations):
        BMR = BlockMixingRotation
        blocks = params['transformer']['h']
        L = len(blocks.keys())
        state = jnp.asarray(checkpoint_state.detach().cpu().numpy())
        forward_fn = BMR._make_forward_fn(state, L)
        backward_fn = BMR._make_backward_fn(state, L)
        return state, forward_fn, backward_fn

    @staticmethod
    def build(params, rng):
        BMR = BlockMixingRotation
        blocks = params['transformer']['h']
        L = len(blocks.keys())
        rngs = jax.random.split(rng, L)
        state = jnp.stack([_gaussian_orthogonal_matrix(L, L, rngs[i]) for i in range(len(BMR._mappers))])
        forward_fn = BMR._make_forward_fn(state, L)
        backward_fn = BMR._make_backward_fn(state, L)
        return state, forward_fn, backward_fn

    @staticmethod
    def _make_forward_fn(state, L):
        BMR = BlockMixingRotation
        def forward(params):
            new_params = copy.deepcopy(params)
            for i, (mapper, setter) in enumerate(zip(BMR._mappers, BMR._setters)):
                stacked_block = jnp.stack([mapper(params['transformer']['h'][str(j)]) for j in range(L)])
                stacked_block = (state[i] @ stacked_block)
                for j in range(L):
                    setter(new_params['transformer']['h'][str(j)], stacked_block[i])
            return new_params
        return forward

    @staticmethod
    def _make_backward_fn(state, L):
        BMR = BlockMixingRotation
        def backward(params):
            new_params = copy.deepcopy(params)
            for i, (mapper, setter) in enumerate(zip(BMR._mappers, BMR._setters)):
                stacked_block = jnp.stack([mapper(params['transformer']['h'][str(j)]) for j in range(L)])
                stacked_block = (state[i].T @ stacked_block)
                for j in range(L):
                    setter(new_params['transformer']['h'][str(j)], stacked_block[i])
            return new_params
        return backward


class LayerNormMixingRotation(BaseRotation):

    _mappers = [
        lambda block: block['ln_1']['scale'],
        lambda block: block['ln_2']['scale'],
    ]
    _setters = [
        lambda block, value: _setter(block, ['ln_1', 'scale'], value),
        lambda block, value: _setter(block, ['ln_2', 'scale'], value),
    ]

    @staticmethod
    def from_checkpoint(checkpoint_state, params, rotations):
        LNMR = LayerNormMixingRotation
        blocks = params['transformer']['h']
        L = len(blocks.keys())
        state = jnp.asarray(checkpoint_state.detach().cpu().numpy())
        forward_fn = LNMR._make_forward_fn(state, L)
        backward_fn = LNMR._make_backward_fn(state, L)
        return state, forward_fn, backward_fn

    @staticmethod
    def build(params, rng):
        LNMR = LayerNormMixingRotation
        blocks = params['transformer']['h']
        L = len(blocks.keys())
        ln_size = LNMR._mappers[0]([blocks['0']]).size
        dim = 2 * L * ln_size
        state = _gaussian_orthogonal_matrix(dim, dim, rng)
        forward_fn = LNMR._make_forward_fn(state, L)
        backward_fn = LNMR._make_backward_fn(state, L)
        return state, forward_fn, backward_fn

    @staticmethod
    def _make_forward_fn(state, L):
        LNMR = LayerNormMixingRotation
        def forward(params):
            new_params = copy.deepcopy(params)
            stacked_params = jnp.stack([
                mapper(params['transformer']['h'][str(i)]) for mapper, i in product(LNMR._mappers, range(L))
            ])
            stacked_params = state @ stacked_params
            for j, (setter, i) in enumerate(product(LNMR._setters, range(L))):
                setter(new_params['transformer']['h'][str(i)], stacked_params[j])
            return new_params
        return forward

    @staticmethod
    def _make_backward_fn(state, L):
        LNMR = LayerNormMixingRotation
        def backward(params):
            new_params = copy.deepcopy(params)
            stacked_params = jnp.stack([
                mapper(params['transformer']['h'][str(i)]) for mapper, i in product(LNMR._mappers, range(L))
            ])
            stacked_params = state.T @ stacked_params
            for j, (setter, i) in enumerate(product(LNMR._setters, range(L))):
                setter(new_params['transformer']['h'][str(i)], stacked_params[j])
            return new_params
        return backward


def _setter(block, path, val):
    while len(path) > 1:
        block = block[path.pop(0)]
    block[path[0]] = val


# Building rotations
def _make_orthogonal_buffers(params, rngs, functions):
    def make_buffers(path, param, rng):
        if any(f(tree_path_to_name(path)) for f in functions):
            return _construct_orthogonal_buffers(param, rng)
    return jax.tree_util.tree_map_with_path(make_buffers, params, rngs)


def _construct_orthogonal_buffers(param, rng):
    rngs = jax.random.split(rng, len(param.shape))
    gaussian_rotators = [_gaussian_orthogonal_matrix(d, d, rngs[i]) for i, d in enumerate(param.shape)]
    return gaussian_rotators


def _gaussian_orthogonal_matrix(n, d, rng):
    size = max(n, d)
    gaussian_matrix = jax.random.normal(rng, [size, size])
    gaussian_matrix, r = jla.qr(gaussian_matrix)
    del r
    gaussian_matrix = gaussian_matrix[:n, :d]
    return gaussian_matrix


# Todo: i think we changed the logic for this part in rotations.py, if so, we should change it here as well
def _apply_rotators(rotators, vec, transpose: bool = False):
    shape = [r.shape[0] for r in rotators]
    r1 = rotators[0].T if transpose else rotators[0]
    r2 = rotators[1].T if transpose else rotators[1]
    if len(rotators) == 2:
        return jnp.einsum('ij,jk,kl->il', r1, vec.reshape(shape), r2).reshape(-1)
    else:
        r3 = rotators[2].T if transpose else rotators[2]
        return jnp.einsum('ip,pqr,jq,kr->ijk', r1, vec.reshape(shape), r2, r3).reshape(-1)


def _make_vecs(params):
    vec_1 = []
    vec_2 = []
    for path, param in tu.tree_flatten_with_path(params)[0]:
        n = tree_path_to_name(path)
        if 'lm_head' not in n:
            if n.endswith('c_proj.kernel'):
                vec_2.append(param.reshape(-1))
            else:
                vec_1.append(param.reshape(-1))
    return jnp.concatenate(vec_1), jnp.concatenate(vec_2)


def _make_vecs_with_filter(params, filter_fn):
    vec_inc = [[], []]
    vec_exc = []
    for path, param in tu.tree_flatten_with_path(params)[0]:
        n = tree_path_to_name(path)
        if filter_fn(n):
            if n.endswith('c_proj.kernel'):
                vec_inc[1].append(param.reshape(-1))
            else:
                vec_inc[0].append(param.reshape(-1))
        else:
            vec_exc.append(param.reshape(-1))
    if len(vec_inc[1]) > 0:
        # We have projections, so separating into two groups
        return [jnp.concatenate(vec_inc[0]), jnp.concatenate(vec_inc[1])], jnp.concatenate(vec_exc)
    else:
        return [jnp.concatenate(vec_inc[0])], jnp.concatenate(vec_exc)


def _reconstruct_params(vecs, params):
    leaves, treedef = tu.tree_flatten(params)
    vec_1, vec_2 = vecs
    vec_1_idx = 0
    vec_2_idx = 0
    new_leaves = []
    for path, param in tu.tree_flatten_with_path(params)[0]:
        n = tree_path_to_name(path)
        if 'lm_head' not in n:
            if n.endswith('c_proj.kernel'):
                size = param.size
                new_leaves.append(vec_2[vec_2_idx:vec_2_idx + size].reshape(param.shape))
                vec_2_idx += size
            else:
                size = param.size
                new_leaves.append(vec_1[vec_1_idx:vec_1_idx + size].reshape(param.shape))
                vec_1_idx += size
        else:
            new_leaves.append(param)
    return tu.tree_unflatten(treedef, new_leaves)


def _reconstruct_params_with_filter(vecs_inc, params, filter_fn):
    leaves, treedef = tu.tree_flatten(params)
    vec_1_idx = 0
    vec_2_idx = 0
    new_leaves = []
    for path, param in tu.tree_flatten_with_path(params)[0]:
        n = tree_path_to_name(path)
        if filter_fn(n):
            if n.endswith('c_proj.kernel'):
                size = param.size
                new_leaves.append(vecs_inc[1][vec_2_idx:vec_2_idx + size].reshape(param.shape))
                vec_2_idx += size
            else:
                size = param.size
                new_leaves.append(vecs_inc[0][vec_1_idx:vec_1_idx + size].reshape(param.shape))
                vec_1_idx += size
        else:
            new_leaves.append(param)
    return tu.tree_unflatten(treedef, new_leaves)
