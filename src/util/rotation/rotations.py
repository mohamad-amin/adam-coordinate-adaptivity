from itertools import product

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Dict

from src.util.rotation.rotation_utils import construct_orthogonal_rotators, gaussian_orthogonal_matrix
from src.model.model import MLP, CausalSelfAttention, LayerNorm


WEIGHT_MAP = {
    'mlp': lambda name: '.mlp.' in name and 'gelu' not in name and 'dropout' not in name,
    'attention': lambda name: '.attn.' in name and 'dropout' not in name,
    'embedding': lambda name: '.wte.' in name or '.wpe.' in name,
    'layernorm': lambda name: '.ln_' in name,
    'secondlayernorm': lambda name: '.ln_2.' in name,
    'firstlayernorm': lambda name: '.ln_1.' in name,
}

FACTORIZATION_MAP = {
    'mlp': [[512, 384, 144], [512, 384, 144]],
    'attention': [[512, 384, 108], [256, 384, 72]],
    'embedding': [[384, 401, 256]],
    'layernorm': [[128, 150]],
    'secondlayernorm': [[128, 72]],
    'firstlayernorm': [[128, 72]],
}


class BaseRotation(object):

    def forward(self, model):
        pass

    def backward(self, model):
        pass

    def backward_weight(self, model):
        pass

    def dump_state(self):
        return None


class Rotator(object):

    def __init__(self, model, rotations):
        self.model = model
        self.rotations = rotations

    @staticmethod
    def Compose(model, rotations, states=None):
        if states is None:
            return Rotator(model, [rotation(model) for i, rotation in enumerate(rotations)])
        else:
            return Rotator(model, [rotation(model, state=states[i]) for i, rotation in enumerate(rotations)])

    def forward(self):
        for rotation in self.rotations:
            rotation.forward(self.model)

    def backward(self):
        for rotation in reversed(self.rotations):
            rotation.backward(self.model)

    def backward_weight(self):
        for rotation in reversed(self.rotations):
            rotation.backward_weight(self.model)

    def dump_state(self):
        return [rotator.dump_state() for rotator in self.rotations]


class WeightRotation(BaseRotation):

    def __init__(self, model, weight_types, state=None):
        self.functions = list(map(lambda wt: WEIGHT_MAP[wt], weight_types))
        self._register_orthogonal_buffers(model)

    def _register_orthogonal_buffers(self, module):
        for name, module in _get_leaf_modules(module):
            if any(f(name) for f in self.functions):
                _register_weight_buffers(module)

    def forward(self, model):
        for name, module in _get_leaf_modules(model):
            if any(f(name) for f in self.functions):
                _forward_weight_buffers(module)

    def backward(self, model):
        for name, module in _get_leaf_modules(model):
            if any(f(name) for f in self.functions):
                _backward_grad_buffers(module)

    def backward_weight(self, model):
        for name, module in _get_leaf_modules(model):
            if any(f(name) for f in self.functions):
                _backward_weight_buffers(module)


def _apply_rotators(rotators: List[torch.Tensor], vec: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    shape = [r.shape[0] for r in rotators]
    vec = vec.reshape(shape)
    r1 = rotators[0].T if transpose else rotators[0]
    r2 = rotators[1].T if transpose else rotators[1]
    if len(rotators) == 2:
        return torch.einsum('ij,jk,kl->il', r1, vec, r2).reshape(-1)
    else:
        r3 = rotators[2].T if transpose else rotators[2]
        return torch.einsum('ip,pqr,jq,kr->ijk', r1, vec, r2, r3).reshape(-1)


def forward_impl(vecs: List[torch.Tensor], perms: List[List[torch.Tensor]], rotators: List[List[List[torch.Tensor]]], k: int) -> List[torch.Tensor]:
    num_vecs = len(vecs)
    for i in range(k):
        for j in range(num_vecs):
            perm = perms[i][j]
            orthogonal_rotators = rotators[i][j]
            vecs[j] = vecs[j][perm]
            vecs[j] = _apply_rotators(orthogonal_rotators, vecs[j])
    return vecs


def backward_impl(vecs: List[torch.Tensor], inverse_perms: List[List[torch.Tensor]], rotators: List[List[List[torch.Tensor]]], k: int) -> List[torch.Tensor]:
    num_vecs = len(vecs)
    for i in range(k-1, -1, -1):
        for j in range(num_vecs):
            perm = inverse_perms[i][j]
            orthogonal_rotators = rotators[i][j]
            vecs[j] = _apply_rotators(orthogonal_rotators, vecs[j], transpose=True)[perm]
    return vecs


def backward_weight_impl(vecs: List[torch.Tensor], inverse_perms: List[List[torch.Tensor]], rotators: List[List[List[torch.Tensor]]], k: int) -> List[torch.Tensor]:
    num_vecs = len(vecs)
    for i in range(k-1, -1, -1):
        for j in range(num_vecs):
            perm = inverse_perms[i][j]
            orthogonal_rotators = rotators[i][j]
            vecs[j] = _apply_rotators(orthogonal_rotators, vecs[j], transpose=True)[perm]
    return vecs


scripted_forward_impl = torch.jit.script(forward_impl)
scripted_backward_impl = torch.jit.script(backward_impl)
scripted_backward_weight_impl = torch.jit.script(backward_weight_impl)


class RandomPermuteRotation(BaseRotation):

    def __init__(self, model, k: int, state=None):
        params_dict = {name: param for name, param in model.named_parameters()}
        device = params_dict[list(params_dict.keys())[0]].device
        vec_1, vec_2 = _make_vecs(model)
        self.factorizations = [[883, 139, 725], [256, 256, 540]]
        self.ps = vec_1.shape[0], vec_2.shape[0]
        self.k = k
        if state is None:
            self.perms = [[torch.randperm(p).to(device) for p in self.ps] for _ in range(k)]
            self.rotators = [[
                [gaussian_orthogonal_matrix(d, d, device=device) for d in fact] for fact in self.factorizations
            ] for _ in range(k)]
            self.state = [self.perms, self.rotators]
        else:
            self.perms = state[0]
            self.rotators = state[1]
            self.state = state
        self.inverse_perms = [[torch.empty_like(self.perms[i][j]) for j in range(2)] for i in range(k)]
        for i in range(k):
            for j in range(2):
                self.inverse_perms[i][j][self.perms[i][j]] = torch.arange(self.perms[i][j].size(0), device=device)

    def forward(self, model):
        with torch.no_grad():
            vecs = list(_make_vecs(model))
            vecs = scripted_forward_impl(vecs, self.perms, self.rotators, self.k)
            _transfer_to_model(model, vecs[0], vecs[1])

    def forward_params(self, params):
        with torch.no_grad():
            vecs = list(_make_vecs_from_params(params))
            vecs = scripted_forward_impl(vecs, self.perms, self.rotators, self.k)
            return _make_params_from_vecs(vecs[0], vecs[1], params, False)

    def backward(self, model):
        with torch.no_grad():
            vecs = list(_make_vecs(model, grads=True))
            vecs = scripted_backward_impl(vecs, self.inverse_perms, self.rotators, self.k)
            _transfer_to_model(model, vecs[0], vecs[1], grads=True)

    def backward_params(self, params):
        with torch.no_grad():
            vecs = list(_make_vecs_from_params(params))
            vecs = scripted_backward_impl(vecs, self.perms, self.rotators, self.k)
            return _make_params_from_vecs(vecs[0], vecs[1], params, False)

    def backward_weight(self, model):
        with torch.no_grad():
            vecs = list(_make_vecs(model))
            vecs = scripted_backward_weight_impl(vecs, self.inverse_perms, self.rotators, self.k)
            _transfer_to_model(model, vecs[0], vecs[1])

    def dump_state(self):
        return self.state


class WeightRandPermRotation(BaseRotation):

    def __init__(self, model, weight_types, k: int, state=None):
        # functions = list(map(lambda wt: WEIGHT_MAP[wt], weight_types))
        # self.filter_fn = lambda n: any(f(n) for f in functions)
        # Todo: only supporting one weight type
        self.filter_fn = WEIGHT_MAP[weight_types[0]]
        self.factorizations = FACTORIZATION_MAP[weight_types[0]]
        params_dict = {name: param for name, param in model.named_parameters()}
        device = params_dict[list(params_dict.keys())[0]].device
        vecs_inc, _ = _make_vecs_with_filter(model, filter_fn=self.filter_fn)
        self.ps = [v.shape[0] for v in vecs_inc]  # Todo: fix, embedding rotation will not work
        self.k = k
        if state is None:
            self.perms = [[torch.randperm(p).to(device) for p in self.ps] for _ in range(k)]
            self.rotators = [[
                [gaussian_orthogonal_matrix(d, d, device=device) for d in fact] for fact in self.factorizations
            ] for _ in range(k)]
            self.state = [self.perms, self.rotators]
        else:
            self.perms = state[0]
            self.rotators = state[1]
            self.state = state
        self.inverse_perms = [[torch.empty_like(self.perms[i][j]) for j in range(len(self.ps))] for i in range(k)]
        for i in range(k):
            for j in range(len(self.ps)):
                self.inverse_perms[i][j][self.perms[i][j]] = torch.arange(self.perms[i][j].size(0), device=device)

    def forward(self, model):
        with torch.no_grad():
            vecs_inc = list(_make_vecs_with_filter(model, self.filter_fn)[0])
            vecs_inc = scripted_forward_impl(vecs_inc, self.perms, self.rotators, self.k)
            _transfer_to_model_with_filter(model, vecs_inc, self.filter_fn)

    def backward(self, model):
        with torch.no_grad():
            vecs_inc = list(_make_vecs_with_filter(model, self.filter_fn, grads=True)[0])
            vecs_inc = scripted_backward_impl(vecs_inc, self.inverse_perms, self.rotators, self.k)
            _transfer_to_model_with_filter(model, vecs_inc, self.filter_fn, grads=True)

    def backward_weight(self, model):
        with torch.no_grad():
            vecs_inc = list(_make_vecs_with_filter(model, self.filter_fn)[0])
            vecs_inc = scripted_backward_weight_impl(vecs_inc, self.inverse_perms, self.rotators, self.k)
            _transfer_to_model_with_filter(model, vecs_inc, self.filter_fn)

    def dump_state(self):
        return self.state


# Assumes the input weight has the correct shape, returns the rotated weight in the same shape as input weight
def _apply_individual_rotators(rotators: List[torch.Tensor], weight: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    r1 = rotators[0].T if transpose else rotators[0]
    if len(rotators) == 1:
        return r1 @ weight

    r2 = rotators[1].T if transpose else rotators[1]
    if len(rotators) == 2:
        return torch.einsum('ij,jk,kl->il', r1, weight, r2)

    r3 = rotators[2].T if transpose else rotators[2]
    if len(rotators) == 3:
        return torch.einsum('ip,pqr,jq,kr->ijk', r1, weight, r2, r3)

    raise ValueError("The length of rotators must be 1, 2, or 3")


EMBEDDING_ORIG_DIM = [50304, 768]
EMBEDDING_GHOST_DIM = [128, 393, 768]


def individual_forward_impl(
        params: Dict[str, torch.Tensor],
        state: List[Dict[str, Tuple[torch.Tensor, List[torch.Tensor]]]],
        inverse_perms: List[Dict[str, torch.Tensor]],
        k: int
) -> Dict[str, torch.Tensor]:
    EMBEDDING_ORIG_DIM = [50304, 768]
    EMBEDDING_GHOST_DIM = [128, 393, 768]
    for i in range(k):
        for name in params.keys():
            if '.ln_f.' not in name:
                continue
            perm, rotators = state[i][name]
            weight = params[name].data
            if '.wte.' in name:
                weight = weight.reshape(-1)[perm].reshape(EMBEDDING_GHOST_DIM)
                params[name].data.copy_(_apply_individual_rotators(rotators, weight).reshape(EMBEDDING_ORIG_DIM))
            else:
                shape = weight.shape
                weight = weight.reshape(-1)[perm].reshape(shape)
                params[name].data.copy_(_apply_individual_rotators(rotators, weight))
    return params


def individual_backward_impl(
        params: Dict[str, torch.Tensor],
        state: List[Dict[str, Tuple[torch.Tensor, List[torch.Tensor]]]],
        inverse_perms: List[Dict[str, torch.Tensor]],
        k: int
) -> Dict[str, torch.Tensor]:
    EMBEDDING_ORIG_DIM = [50304, 768]
    EMBEDDING_GHOST_DIM = [128, 393, 768]
    for i in range(k):
        for name in params.keys():
            if '.ln_f.' not in name:
                continue
            inverse_perm, rotators = inverse_perms[i][name], state[i][name][1]
            if '.wte.' in name:
                grad = _apply_individual_rotators(rotators, params[name].grad.reshape(EMBEDDING_GHOST_DIM), transpose=True)
                grad = grad.reshape(EMBEDDING_ORIG_DIM).reshape(-1)[inverse_perm].reshape(EMBEDDING_ORIG_DIM)
                params[name].grad.copy_(grad)
            else:
                grad = _apply_individual_rotators(rotators, params[name].grad, transpose=True)
                shape = grad.shape
                grad = grad.reshape(-1)[inverse_perm].reshape(shape)
                params[name].grad.copy_(grad)
    return params


def individual_backward_weight_impl(
        params: Dict[str, torch.Tensor],
        state: List[Dict[str, Tuple[torch.Tensor, List[torch.Tensor]]]],
        inverse_perms: List[Dict[str, torch.Tensor]],
        k: int
) -> Dict[str, torch.Tensor]:
    EMBEDDING_ORIG_DIM = [50304, 768]
    EMBEDDING_GHOST_DIM = [128, 393, 768]
    for i in range(k):
        for name in params.keys():
            if '.ln_f.' not in name:
                continue
            inverse_perm, rotators = inverse_perms[i][name], state[i][name][1]
            if '.wte.' in name:
                weight = _apply_individual_rotators(rotators, params[name].data.reshape(EMBEDDING_GHOST_DIM), transpose=True)
                weight = weight.reshape(EMBEDDING_ORIG_DIM).reshape(-1)[inverse_perm].reshape(EMBEDDING_ORIG_DIM)
                params[name].data.copy_(weight)
            else:
                weight = _apply_individual_rotators(rotators, params[name].data, transpose=True)
                shape = weight.shape
                weight = weight.reshape(-1)[inverse_perm].reshape(shape)
                params[name].data.copy_(weight)
    return params


scripted_individual_forward_impl = torch.jit.script(individual_forward_impl)
scripted_individual_backward_impl = torch.jit.script(individual_backward_impl)
scripted_individual_backward_weight_impl = torch.jit.script(individual_backward_weight_impl)


class IndividualRandPermRotation(BaseRotation):

    def __init__(self, model, k: int, state=None):

        params_dict = self._get_params_dict(model)
        device = params_dict[list(params_dict.keys())[0]].device

        if state is None:
            states = [{} for _ in range(k)]

            for i in range(k):
                state = states[i]
                for name, param in params_dict.items():
                    if '.ln_f.' not in name:
                        continue
                    m = param.numel()
                    perm = torch.randperm(m).to(device)
                    if '.wte.' not in name:
                        rotators = construct_orthogonal_rotators(param)
                    else:
                        rotators = construct_orthogonal_rotators(param.data.view(EMBEDDING_GHOST_DIM))
                    state[name] = (perm, rotators)

            self.state = states
        else:
            self.state = state

        self.k = k
        self.inverse_perms_state = [{} for _ in range(k)]
        for i in range(k):
            inverse_perms = self.inverse_perms_state[i]
            for name, _ in params_dict.items():
                if '.ln_f.' not in name:
                    continue
                inverse_perms[name] = torch.argsort(self.state[i][name][0])

    def _get_params_dict(self, model):
        return {name: param for name, param in model.named_parameters()}

    def forward(self, model):
        with torch.no_grad():
            params_dict = self._get_params_dict(model)
            scripted_individual_forward_impl(params_dict, self.state, self.inverse_perms_state, self.k)

    def backward(self, model):
        with torch.no_grad():
            params_dict = self._get_params_dict(model)
            scripted_individual_backward_impl(params_dict, self.state, self.inverse_perms_state, self.k)

    def backward_weight(self, model):
        with torch.no_grad():
            params_dict = self._get_params_dict(model)
            scripted_individual_backward_weight_impl(params_dict, self.state, self.inverse_perms_state, self.k)

    def dump_state(self):
        return self.state



class BlockMixingRotation(BaseRotation):

    def __init__(self, model, state=None):
        self.n_layers = len(model.transformer.h)
        self.mappers = [
            lambda block: block.ln_1,
            lambda block: block.attn.c_attn,
            lambda block: block.attn.c_proj,
            lambda block: block.ln_2,
            lambda block: block.mlp.c_fc,
            lambda block: block.mlp.c_proj
        ]
        if state is None:
            self.state = torch.stack([
                gaussian_orthogonal_matrix(self.n_layers, self.n_layers, device=next(model.parameters()).device)
                for _ in range(len(self.mappers))
            ]).contiguous()
        else:
            self.state = state

    def forward(self, model):
        for i, mapper in enumerate(self.mappers):
            with torch.no_grad():
                # L x ...
                L = self.state[i].shape[0]
                stacked_weights = torch.stack([mapper(block).weight.data for block in model.transformer.h], dim=0)
                stacked_weights = (self.state[i] @ stacked_weights.view(L, -1)).view(stacked_weights.shape)
                for j, block in enumerate(model.transformer.h):
                    mapper(block).weight.data = stacked_weights[j]

    def backward(self, model):
        for i, mapper in enumerate(self.mappers):
            with torch.no_grad():
                # L x ...
                L = self.state[i].shape[0]
                stacked_grads = torch.stack([mapper(block).weight.grad for block in model.transformer.h], dim=0)
                stacked_grads = (self.state[i].T @ stacked_grads.view(L, -1)).view(stacked_grads.shape)
                for i, block in enumerate(model.transformer.h):
                    mapper(block).weight.grad = stacked_grads[i]

    def backward_weight(self, model):
        for i, mapper in enumerate(self.mappers):
            with torch.no_grad():
                # L x ...
                L = self.state[i].shape[0]
                stacked_weights = torch.stack([mapper(block).weight.data for block in model.transformer.h], dim=0)
                stacked_weights = (self.state[i].T @ stacked_weights.view(L, -1)).view(stacked_weights.shape)
                for i, block in enumerate(model.transformer.h):
                    mapper(block).weight.data = stacked_weights[i]

    def dump_state(self):
        return self.state


class LayerNormMixingRotation(BaseRotation):

    def __init__(self, model, state=None):
        self.n_layers = len(model.transformer.h)
        self.mappers = [
            lambda block: block.ln_1,
            lambda block: block.ln_2
        ]
        ln_size = len(self.mappers[0](model.transformer.h[0]).weight)
        self.dim = 2 * ln_size * self.n_layers
        if state is None:
            self.state = gaussian_orthogonal_matrix(self.dim, self.dim, device=next(model.parameters()).device).contiguous()
        else:
            self.state = state

    def forward(self, model):
        with torch.no_grad():
            stacked_weights = torch.stack([
                mapper(block).weight.data for mapper, block in product(self.mappers, model.transformer.h)])
            stacked_weights = (self.state @ stacked_weights.view(self.dim, -1)).view(stacked_weights.shape)
            for i, (mapper, block) in enumerate(product(self.mappers, model.transformer.h)):
                mapper(block).weight.data = stacked_weights[i]

    def backward(self, model):
        with torch.no_grad():
            stacked_grads = torch.stack([
                mapper(block).weight.grad for mapper, block in product(self.mappers, model.transformer.h)])
            stacked_grads = (self.state.T @ stacked_grads.view(self.dim, -1)).view(stacked_grads.shape)
            for i, (mapper, block) in enumerate(product(self.mappers, model.transformer.h)):
                mapper(block).weight.data = stacked_grads[i]

    def backward_weight(self, model):
        with torch.no_grad():
            stacked_weights = torch.stack([
                mapper(block).weight.data for mapper, block in product(self.mappers, model.transformer.h)])
            stacked_weights = (self.state.T @ stacked_weights.view(self.dim, -1)).view(stacked_weights.shape)
            for i, (mapper, block) in enumerate(product(self.mappers, model.transformer.h)):
                mapper(block).weight.data = stacked_weights[i]

    def dump_state(self):
        return self.state


class EmbeddingRotation(BaseRotation):

    def __init__(self, model):
        self._register_orthogonal_buffers(model)

    def _register_orthogonal_buffers(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Embedding):
                _register_weight_buffers(child)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, MLP, CausalSelfAttention, LayerNorm)):
                pass
            else:
                self._register_orthogonal_buffers(child)

    def forward(self, model):
        for name, child in model.named_children():
            if isinstance(child, nn.Embedding):
                _forward_weight_buffers(child)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, MLP, CausalSelfAttention, LayerNorm)):
                pass
            else:
                self.forward(child)

    def backward(self, model):
        for name, child in model.named_children():
            if isinstance(child, nn.Embedding):
                _backward_grad_buffers(child)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, MLP, CausalSelfAttention, LayerNorm)):
                pass
            else:
                self.backward(child)

    def backward_weight(self, model):
        for name, child in model.named_children():
            if isinstance(child, nn.Embedding):
                _backward_weight_buffers(child)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, MLP, CausalSelfAttention, LayerNorm)):
                pass
            else:
                self.backward_weight(child)


class MLPRotation(BaseRotation):

    def __init__(self, model):
        self._register_orthogonal_buffers(model)

    def _register_orthogonal_buffers(self, module):
        for name, child in module.named_children():
            if isinstance(child, MLP):
                _register_weight_buffers(child.c_fc)
                _register_weight_buffers(child.c_proj)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, CausalSelfAttention, LayerNorm)):
                pass
            else:
                self._register_orthogonal_buffers(child)

    def forward(self, model):
        for name, child in model.named_children():
            if isinstance(child, MLP):
                _forward_weight_buffers(child.c_fc)
                _forward_weight_buffers(child.c_proj)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, CausalSelfAttention, LayerNorm)):
                pass
            else:
                self.forward(child)

    def backward(self, model):
        for name, child in model.named_children():
            if isinstance(child, MLP):
                _backward_grad_buffers(child.c_fc)
                _backward_grad_buffers(child.c_proj)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, CausalSelfAttention, LayerNorm)):
                pass
            else:
                self.backward(child)

    def backward_weight(self, model):
        for name, child in model.named_children():
            if isinstance(child, MLP):
                _backward_weight_buffers(child.c_fc)
                _backward_weight_buffers(child.c_proj)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, CausalSelfAttention, LayerNorm)):
                pass
            else:
                self.backward_weight(child)


class AttentionRotation(BaseRotation):

    def __init__(self, model):
        self._register_orthogonal_buffers(model)

    def _register_orthogonal_buffers(self, module):
        for name, child in module.named_children():
            if isinstance(child, CausalSelfAttention):
                _register_weight_buffers(child.c_proj)
                _register_weight_buffers(child.c_attn)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, MLP, LayerNorm)):
                pass
            else:
                self._register_orthogonal_buffers(child)

    def forward(self, model):
        for name, child in model.named_children():
            if isinstance(child, CausalSelfAttention):
                _forward_weight_buffers(child.c_proj)
                _forward_weight_buffers(child.c_attn)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, MLP, LayerNorm)):
                pass
            else:
                self.forward(child)

    def backward(self, model):
        for name, child in model.named_children():
            if isinstance(child, CausalSelfAttention):
                _backward_grad_buffers(child.c_proj)
                _backward_grad_buffers(child.c_attn)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, MLP, LayerNorm)):
                pass
            else:
                self.backward(child)

    def backward_weight(self, model):
        for name, child in model.named_children():
            if isinstance(child, CausalSelfAttention):
                _backward_weight_buffers(child.c_proj)
                _backward_weight_buffers(child.c_attn)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, MLP, LayerNorm)):
                pass
            else:
                self.backward_weight(child)


class LayerNormRotation(BaseRotation):

    def __init__(self, model):
        self._register_orthogonal_buffers(model)

    def _register_orthogonal_buffers(self, module):
        for name, child in module.named_children():
            if isinstance(child, LayerNorm):
                _register_weight_buffers(child)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, MLP, CausalSelfAttention)):
                pass
            else:
                self._register_orthogonal_buffers(child)

    def forward(self, model):
        for name, child in model.named_children():
            if isinstance(child, LayerNorm):
                _forward_weight_buffers(child)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, MLP, CausalSelfAttention)):
                pass
            else:
                self.forward(child)

    def backward(self, model):
        for name, child in model.named_children():
            if isinstance(child, LayerNorm):
                _backward_grad_buffers(child)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, MLP, CausalSelfAttention)):
                pass
            else:
                self.backward(child)

    def backward_weight(self, model):
        for name, child in model.named_children():
            if isinstance(child, LayerNorm):
                _backward_weight_buffers(child)
            elif isinstance(child, (nn.GELU, nn.Dropout, nn.Linear, nn.Embedding, MLP, CausalSelfAttention)):
                pass
            else:
                self.backward_weight(child)


class LastLayerNormRotation(BaseRotation):

    def __init__(self, model):
        self._register_orthogonal_buffers(model)

    def _register_orthogonal_buffers(self, module):
        _register_weight_buffers(module.transformer.ln_f)

    def forward(self, model):
        _forward_weight_buffers(model.transformer.ln_f)

    def backward(self, model):
        _backward_grad_buffers(model.transformer.ln_f)

    def backward_weight(self, model):
        _backward_weight_buffers(model.transformer.ln_f)


# Can we do this in a cleaner way? I don't like writing function in the bare file like this.
def _register_weight_buffers(module):
    w = module.weight
    if not hasattr(module, '_w_u0'):
        for i, u in enumerate(construct_orthogonal_rotators(w)):
            module.register_buffer('_w_u' + str(i), u.contiguous())


def _forward_weight_buffers(module):
    w = module.weight
    # for d in range(len(w.shape)):
    #     u = getattr(module, '_w_u' + str(d))
    #     with torch.no_grad():
    #         w.data = torch.tensordot(w.data, u, dims=([d], [0])).reshape(w.shape)
    u1 = module._w_u0
    if len(w.shape) == 1:
        with torch.no_grad():
            w.data = u1 @ w.data
    else:
        u2 = module._w_u1
        with torch.no_grad():
            w.data = u1 @ w.data @ u2


def _backward_weight_buffers(module):
    w = module.weight
    u1 = module._w_u0
    if len(w.shape) == 1:
        with torch.no_grad():
            w.data = u1.T @ w.data
    else:
        u2 = module._w_u1
        with torch.no_grad():
            w.data = u1.T @ w.data @ u2.T


def _backward_grad_buffers(module):
    w = module.weight
    u1 = module._w_u0
    if len(w.shape) == 1:
        with torch.no_grad():
            w.grad = u1.T @ w.grad
    else:
        u2 = module._w_u1
        with torch.no_grad():
            w.grad = u1.T @ w.grad @ u2.T


def _get_leaf_modules(module, parent_name=''):
    leaf_modules = []
    for name, child in module.named_children():
        full_name = f'{parent_name}.{name}' if parent_name else name
        # Check if the child has further children
        if any(child.named_children()):
            leaf_modules.extend(_get_leaf_modules(child, full_name))
        else:
            leaf_modules.append((full_name, child))
    return leaf_modules


def _make_vecs(model, grads=False):
    vec_1 = []
    vec_2 = []
    for n, p in model.named_parameters():
        if 'lm_head' not in n:
            if n.endswith('c_proj.weight'):
                vec_2.append(p.grad.view(-1) if grads else p.view(-1))
            else:
                vec_1.append(p.grad.view(-1) if grads else p.view(-1))
    return torch.cat(vec_1), torch.cat(vec_2)

def _make_vecs_from_params(params, grads=False):
    vec_1 = []
    vec_2 = []
    for n, p in params.items():
        if 'lm_head' not in n:
            if n.endswith('c_proj.weight'):
                vec_2.append(p.grad.view(-1) if grads else p.view(-1))
            else:
                vec_1.append(p.grad.view(-1) if grads else p.view(-1))
    return torch.cat(vec_1), torch.cat(vec_2)


def _make_params_from_vecs(vec_1, vec_2, original_params, grads=False):
    params = {}
    idx_1 = 0
    idx_2 = 0

    for n, p in original_params.items():
        if 'lm_head' not in n:
            if n.endswith('c_proj.weight'):
                length = p.numel()
                flat_tensor = vec_2[idx_2:idx_2 + length]
                tensor = flat_tensor.view_as(p)
                if grads:
                    p.grad = tensor
                else:
                    p.data = tensor
                idx_2 += length
            else:
                length = p.numel()
                flat_tensor = vec_1[idx_1:idx_1 + length]
                tensor = flat_tensor.view_as(p)
                if grads:
                    p.grad = tensor
                else:
                    p.data = tensor
                idx_1 += length
            params[n] = p
    return params


def _make_vecs_with_filter(model, filter_fn, grads=False):
    vec_inc = [[], []]
    vec_exc = []
    for n, p in model.named_parameters():
        if filter_fn(n):
            if n.endswith('c_proj.weight'):
                vec_inc[1].append(p.grad.view(-1) if grads else p.view(-1))
            else:
                vec_inc[0].append(p.grad.view(-1) if grads else p.view(-1))
        else:
            vec_exc.append(p.grad.view(-1) if grads else p.view(-1))
    if len(vec_inc[1]) > 0:
        # We have projections, so separating into two groups
        return [torch.cat(vec_inc[0]), torch.cat(vec_inc[1])], torch.cat(vec_exc)
    else:
        return [torch.cat(vec_inc[0])], torch.cat(vec_exc)


def _transfer_to_model(model, vec_1, vec_2, grads=False):
    p1, p2 = 0, 0
    for n, p in model.named_parameters():
        if 'lm_head' not in n:
            numel = p.numel()
            if n.endswith('c_proj.weight'):
                if grads:
                    p.grad.copy_(vec_2[p2:p2+numel].view_as(p))
                else:
                    p.data.copy_(vec_2[p2:p2+numel].view_as(p))
                p2 += numel
            else:
                if grads:
                    p.grad.copy_(vec_1[p1:p1+numel].view_as(p))
                else:
                    p.data.copy_(vec_1[p1:p1+numel].view_as(p))
                p1 += numel


def _transfer_to_model_with_filter(model, vec_inc, filter_fn, grads=False):
    p1, p2 = 0, 0
    for n, p in model.named_parameters():
        numel = p.numel()
        if filter_fn(n):
            if n.endswith('c_proj.weight'): # If there's nothing, this part just never gets called
                if grads:
                    p.grad.copy_(vec_inc[1][p2:p2 + numel].view_as(p))
                else:
                    p.data.copy_(vec_inc[1][p2:p2 + numel].view_as(p))
                p2 += numel
            else:
                if grads:
                    p.grad.copy_(vec_inc[0][p1:p1 + numel].view_as(p))
                else:
                    p.data.copy_(vec_inc[0][p1:p1 + numel].view_as(p))
                p1 += numel
