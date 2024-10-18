import os
import time
import pickle
from functools import partial

import optax
from flax.training.common_utils import shard, onehot
from jax.flatten_util import ravel_pytree

from src.engine.base.base_jax_engine import BaseJaxEngine
from src.util.rotation.jax_rotations import *


ROTATION_DICT = {
    'weight': WeightRotation,
    'block_mix': BlockMixingRotation,
    'layer_norm_mix': LayerNormMixingRotation,
}


def create_diagonal_matrix(n, t, lambda_1):
    eigenvalues = jnp.array([lambda_1 * np.exp(-t * i) for i in range(n)])
    diagonal_matrix = jnp.diag(eigenvalues)
    return diagonal_matrix


class HessianEvalEngine(BaseJaxEngine):

    def register_configs(self):
        self.hessian_config = self.config['hessian']
        self.train_config = self.config['train']
        self.eval_config = self.config['eval']

    def prepare(self):
        super().prepare()
        self._prepare_rotations()
        print('Done with rotations')
        self.iter_num = self.checkpoint['iter_num']
        self.snr_dir = os.path.join(self.save_dir, 'snr')
        self.hvp_dir = os.path.join(self.save_dir, 'hvp')
        os.makedirs(self.snr_dir, exist_ok=True)
        os.makedirs(self.hvp_dir, exist_ok=True)

    def _prepare_rotations(self, ):

        def rotation_maker(name, from_checkpoint=False):
            if name.startswith('weightrp'):
                start_idx = name.find('_')
                k = int(name[start_idx + 1:start_idx + 2])
                name = name[start_idx + 2:]
                start_idx = name.find('__')
                args_string = name[start_idx + 2:]
                weight_types = [arg.strip() for arg in args_string.split('.')]
                if from_checkpoint:
                    return partial(WeightRandPermRotation.from_checkpoint, weight_types=weight_types)
                else:
                    return partial(WeightRandPermRotation.build, weight_types=weight_types, k=k)
            elif name.startswith('weight'):
                start_idx = name.find('__')
                args_string = name[start_idx + 2:]
                weight_types = [arg.strip() for arg in args_string.split('.')]
                if from_checkpoint:
                    return partial(WeightRotation.from_checkpoint, weight_types=weight_types)
                else:
                    return partial(WeightRotation.build, weight_types=weight_types)
            elif name.startswith('randperm'):
                start_idx = name.find('_')
                k = int(name[start_idx + 1:])
                if from_checkpoint:
                    return partial(RandomPermuteRotation.from_checkpoint)
                else:
                    return partial(RandomPermuteRotation.build, k=k)
            else:
                if from_checkpoint:
                    return ROTATION_DICT[name].from_checkpoint
                else:
                    return ROTATION_DICT[name].build

        if 'rotations' in self.model_config and self.model_config.get('rotate', False):
            print('Found train rotations')
            fns = recover_rotations([
                rotation_maker(rotation_name, from_checkpoint=True) for rotation_name in self.model_config['rotations']
            ], self.checkpoint['rotations'], self.model.params, self.checkpoint_rotations)
            self.train_forward_fns = fns[0]
            self.train_backward_fns = fns[1]
        else:
            self.train_forward_fns = []
            self.train_backward_fns = []

        if not self.hessian_config.get('rotated_evaluation', False):
            print('ID Evaluation')
            self.forward_fns = []
            self.backward_fns = []
        else:
            if self.hessian_config.get('use_train_rotations', False):
                print('Rotated evaluation using train rotations')
                self.forward_fns = self.train_forward_fns
                self.backward_fns = self.train_backward_fns
            elif len(self.hessian_config.get('rotations', [])) > 0:
                print('Rotated evaluation with custom evaluations')
                if 'seed' in self.hessian_config:
                    seed = self.hessian_config['seed']
                else:
                    seed = self.data_config['seed']
                rng = jax.random.PRNGKey(seed)
                fns = make_rotations(self.model.params, [
                    rotation_maker(rotation_name) for rotation_name in self.hessian_config['rotations']
                ], rng)
                self.forward_fns = fns[0]
                self.backward_fns = fns[1]
            else:
                print('ID Evaluation')
                self.forward_fns = []
                self.backward_fns = []

    def run(self):

        seed = self.data_config['seed']
        batch_size, block_size = self.config['train']['batch']['size'], self.config['train']['batch']['block']
        batch_size *= self.world_size

        rng = jax.random.PRNGKey(seed)
        get_batch = lambda split: self.get_batch(split, batch_size, block_size)
        unravel_fn = fu.ravel_pytree(self.model.params)[1]

        def ce_loss(params, batch):
            labels = batch.pop('labels')
            for forward_fn in self.forward_fns:
                params = forward_fn(params)
            # params = unravel_fn(Q @ fu.ravel_pytree(params)[0])
            logits = self.model(**batch, params=params, dropout_rng=rng, train=True)[0]
            loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])).mean()
            return loss

        def compute_gradients(params, batch):
            loss_fn = partial(ce_loss, batch=batch)
            g = jax.grad(loss_fn, allow_int=True)(params)
            # return jax.lax.pmean(g, axis_name='batch')
            return g

        def compute_hvp(params, v, batch):
            loss_fn = partial(ce_loss, batch=batch)
            def hvp_reverse_over_forward(params, v):
                jvp_fun = lambda params: jax.jvp(loss_fn, (params,), (v,))[1]
                return jax.grad(jvp_fun, allow_int=True)(params)
            return jax.lax.pmean(hvp_reverse_over_forward(params, v), axis_name='batch')

        def compute_jacobian(params, batch):
            loss_fn = partial(ce_loss, batch=batch)
            j = jax.jacobian(loss_fn, allow_int=True)(params)
            return jax.lax.pmean(j, axis_name='batch')

        compute_loss = jax.pmap(ce_loss, in_axes=(None, 0), axis_name='batch')
        # grad_fn = jax.pmap(compute_gradients, in_axes=(None, 0), out_axes=None, axis_name='batch')
        grad_fn = jax.pmap(compute_gradients, in_axes=(None, 0), axis_name='batch')
        jacobian_fn = jax.pmap(compute_jacobian, in_axes=(None, 0), out_axes=None, axis_name='batch')
        # hessian_fn = jax.pmap(compute_hessian, in_axes=(None, 0), out_axes=None, axis_name='batch')
        hvp_fn = jax.pmap(compute_hvp, in_axes=(None, None, 0), out_axes=None, axis_name='batch')
        unravel_fn = ravel_pytree(self.model.params)[1]

        if len(self.backward_fns) > 0:
            # Should invert the params:
            params = self.model.params
            for backward_fn in reversed(self.backward_fns):
                params = backward_fn(params)
        else:
            params = self.model.params

        # params = unravel_fn(Q.T @ fu.ravel_pytree(params)[0])

        if self.writer:
            losses = self._estimate_loss(params, get_batch, compute_loss)
            print(f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        task = self.hessian_config.get('task', 'compute_cauchy_vhps')
        if task == 'compute_eigvals':
            self.compute_eigvals(params, grad_fn, hvp_fn, get_batch, unravel_fn, rng)
        elif task == 'compute_11_norm':
            self.compute_11_norm(params, hvp_fn, get_batch, unravel_fn, rng)
        elif task == 'compute_fro_norm':
            self.compute_frobenius_norm(params, hvp_fn, get_batch, unravel_fn, rng)
        elif task == 'compute_snr':
            self.compute_signal_to_noise_ratio(params, grad_fn, get_batch)
        elif task == 'compute_grad_l1_norm':
            self.compute_grad_l1_norm_info(params, grad_fn, get_batch)
        elif task == 'manual_investigation':
            get_batch_fn = get_batch
            gradient_accumulation_steps = self.config['train']['batch']['gradient_accumulation_steps'] // self.world_size
            self.logger.info('Loading data...')
            batches = []
            for _ in range(gradient_accumulation_steps):
                batches.append(get_batch_fn('train'))
            self.logger.info('Done loading data.')
            import IPython; IPython.embed()

    def _estimate_loss(self, params, get_batch, compute_loss):
        out = {}
        eval_iters = self.eval_config['iters'] // self.world_size
        for split in ['train', 'val']:
            loss = 0.
            for k in range(eval_iters):
                loss += compute_loss(params, get_batch(split)).mean()
            out[split] = loss / eval_iters
        return out

    def compute_stochastic_gradient(self, grad_fn, params, batches):

        steps = len(batches)
        tree_sum = jax.jit(lambda a, b: jax.tree_util.tree_map(lambda x, y: x + y, a, b))
        tree_mean_ravel = jax.jit(lambda t: ravel_pytree(jax.tree_util.tree_map(lambda x: x.mean(axis=0), t))[0])
        grads = grad_fn(params, batches[0])

        t0 = time.time()
        iter_num = 0
        for micro_step in range(1, steps):
            grads = tree_sum(grads, grad_fn(params, batches[micro_step]))
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            iter_num += 1
            if self.writer and iter_num % 100 == 0:
                print(f"iter {iter_num}: time {dt * 1000:.2f}ms")

        return tree_mean_ravel(grads) / steps

    def compute_stochastic_vhp(self, hvp_fn, vector, params, batches):

        steps = len(batches)
        hvp = jax.tree_util.tree_map(jnp.zeros_like, vector)
        tree_sum = jax.jit(lambda a, b: jax.tree_util.tree_map(lambda x, y: x + y, a, b))

        t0 = time.time()
        iter_num = 0
        for micro_step in range(steps):
            hvp = tree_sum(hvp, hvp_fn(params, vector, batches[micro_step]))
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            iter_num += 1
            if self.writer and iter_num % 100 == 0:
                print(f"iter {iter_num}: time {dt * 1000:.2f}ms")

        return jax.tree_util.tree_map(lambda x: x / steps, hvp)

    def compute_eigvals(self, params, grad_fn, hvp_fn, get_batch_fn, unravel_fn, rng):

        def gram_schmidt(A):
            a, b = A.shape
            Q = np.zeros((a, b))
            for i in range(a):
                qi = A[i, :]
                for j in range(i):
                    qi -= np.dot(Q[j, :], A[i, :]) * Q[j, :]
                Q[i, :] = qi / np.linalg.norm(qi)
            return Q

        num_power_iters = self.hessian_config['num_power_iters']
        gradient_accumulation_steps = self.config['train']['batch']['gradient_accumulation_steps'] // self.world_size

        self.logger.info('Loading data...')
        batches = []
        for _ in range(gradient_accumulation_steps):
            batches.append(get_batch_fn('train'))
        self.logger.info('Done loading data.')

        T = np.zeros([num_power_iters, num_power_iters])
        # Starting with the gradient as initial vec
        b = self.compute_stochastic_gradient(grad_fn, params, batches)
        b = b / jla.norm(b)
        tmp_b = self.compute_stochastic_vhp(hvp_fn, unravel_fn(b), params, batches)
        tmp_b = ravel_pytree(tmp_b)[0]
        alpha = jnp.dot(tmp_b, b)
        tmp_b = tmp_b - alpha * b
        T[0, 0] = alpha
        V = [jax.device_get(b)]
        for j in range(1, num_power_iters):
            beta = jla.norm(tmp_b)
            if self.writer:
                eval = jnp.max(jnp.abs(jnp.linalg.eigvalsh(T[:j, :j]))).tolist()
                print('iter', j, 'eigenvalue', eval, 'beta', beta)
            if beta < 1e-10:
                T = T[:j, :j]
                break
            else:
                b_prev = b
                b = tmp_b / beta
                V.append(jax.device_get(b))
                tmp_b = self.compute_stochastic_vhp(hvp_fn, unravel_fn(b), params, batches)
                tmp_b = ravel_pytree(tmp_b)[0]
                if jnp.any(jnp.isnan(tmp_b)):
                    print('NaN tmp_b!')
                alpha = jnp.dot(tmp_b, b)
                T[j, j] = alpha
                T[j, j - 1] = beta
                T[j - 1, j] = beta
                tmp_b = tmp_b - alpha * b - beta * b_prev

        V = np.array(V)
        Q = gram_schmidt(V)
        Tvecs = jnp.linalg.eigh(T)[1]
        vecs = jax.device_get(Q.T @ Tvecs)
        top_vec = vecs[:, -1]

        if self.writer:
            jnp.save(os.path.join(self.hvp_dir, f'evals_T_{num_power_iters}.pt'), T)
            jnp.save(os.path.join(self.hvp_dir, f'evals_top_vec_{num_power_iters}.pt'), top_vec)
            eigenvalue = jnp.max(jnp.abs(jnp.linalg.eigvalsh(T))).tolist()
            jnp.set_printoptions(precision=15)
            print('Max eigval:', eigenvalue, 'beta:', beta)
            results = {"eigenvalue": eigenvalue, "beta": beta}
            with open(os.path.join(self.hvp_dir, f'eigenvalue_{num_power_iters}.pt'), 'wb') as f:
                pickle.dump(results, f)

    def _reshape_batches(self, batches, virtual_batch_size, hessian_batch_size):

        gradient_accumulation_steps = self.config['train']['batch']['gradient_accumulation_steps'] // self.world_size
        block_size = self.config['train']['batch']['block']

        concatenated_batches = {key: np.concatenate([batch[key] for batch in batches], axis=0).reshape(-1, block_size)
                                for key in batches[0].keys()}

        reshaped_batches = []
        num_batches = (gradient_accumulation_steps * virtual_batch_size) // hessian_batch_size

        for i in range(num_batches):
            start = i * hessian_batch_size * self.world_size
            end = start + hessian_batch_size * self.world_size
            new_batch = {key: shard(concatenated_batches[key][start:end]) for key in concatenated_batches.keys()}
            reshaped_batches.append(new_batch)

        return reshaped_batches

    def compute_11_norm(self, params, hvp_fn, get_batch_fn, unravel_fn, rng):

        num_cauchy_vecs = self.hessian_config['num_cauchy_vecs']
        d = sum(p.size for p in jax.tree_util.tree_leaves(params))
        gradient_accumulation_steps = self.config['train']['batch']['gradient_accumulation_steps'] // self.world_size

        self.logger.info('Loading data...')
        batches = []
        for _ in range(gradient_accumulation_steps):
            batches.append(get_batch_fn('train'))
        self.logger.info('Done loading data.')

        # Todo: manual investigation
        # batches = batches[:1]

        hvps = []
        is_not_nan = []

        for j in range(num_cauchy_vecs):

            if self.writer:
                print('Processing vec', j)
            rng, new_rng = jax.random.split(rng)
            cauchy = unravel_fn(jnp.array(np.random.standard_cauchy([d]) / d))

            t0 = time.time()
            hvp = self.compute_stochastic_vhp(hvp_fn, cauchy, params, batches)
            hvp = ravel_pytree(hvp)[0]
            if jnp.any(jnp.isnan(hvp)):
                print('NaN encountered!')
                is_not_nan.append(False)
            else:
                is_not_nan.append(True)

            t1 = time.time()
            dt = t1 - t0
            print(f'Overall time {dt * 1000:.2f}ms')
            hvp = jax.device_get(hvp)
            hvps.append(hvp)

            if self.hessian_config.get('save_cauchy_vecs', False):
                print('No space, not saving cauchy vectors!')
                # print(f"Saving cauchy vec {j} to {os.path.join(self.hvp_dir, f'cauchy_hvp_{j}')}")
                # np.save(os.path.join(self.hvp_dir, f'cauchy_hvp_{j}'), np.asarray(hvp))

        hvps = np.stack(hvps).T
        hvps = hvps[:, is_not_nan]

        if self.writer:
            quantiles = np.quantile(hvps, jnp.array([0.25, 0.75]), axis=1)  # num_param * 2
            print('Quantiles', quantiles[:20, :])
            absolute_medians = np.quantile(np.abs(hvps), np.array([0.5]), axis=1)  # num_param * 2
            print("Quantile shape:", quantiles.shape)
            del hvps
            # estimate by half interquantile (75% quantile - 25% quantile)/2
            estimate = np.sum(quantiles[1, :] - quantiles[0, :]) / 2
            estimate = estimate.tolist()
            operator_norm = np.max(quantiles[1, :] - quantiles[0, :]) / 2  # largest row sum of Hessian
            operator_norm = operator_norm.tolist()
            # estimate by median of the absolute value
            estimate_median = np.sum(absolute_medians)
            estimate_median = estimate_median.tolist()
            operator_norm_median = np.max(absolute_medians)
            operator_norm_median = operator_norm_median.tolist()
            np.set_printoptions(precision=15)
            # here l11 norm is divided by num of params
            print(estimate, estimate * d, estimate_median, estimate_median * d)
            results = {"l11 norm": estimate, 'l11 norm median': estimate_median, 'operator': operator_norm,
                       'operator median': operator_norm_median, 'iter_num': self.iter_num,
                       'num_cauchy_vecs': num_cauchy_vecs}
            with open(os.path.join(self.hvp_dir, f'11norm_info.pt'), 'wb') as f:
                pickle.dump(results, f)

    def compute_frobenius_norm(self, params, hvp_fn, get_batch_fn, unravel_fn, rng):

        num_fro_vecs = self.hessian_config['num_fro_vecs']
        gradient_accumulation_steps = self.config['train']['batch']['gradient_accumulation_steps'] // self.world_size
        d = sum(p.size for p in jax.tree_util.tree_leaves(params))

        self.logger.info('Loading data...')
        batches = []
        for _ in range(gradient_accumulation_steps):
            batches.append(get_batch_fn('train'))
        gradient_accumulation_steps = len(batches)
        self.logger.info('Done loading data.')

        hvps_norms = []

        for j in range(num_fro_vecs):

            if self.writer:
                print('Processing vec', j)
            rng, new_rng = jax.random.split(rng)

            vec = unravel_fn(jax.random.choice(new_rng, jnp.array([-1., 1.]), shape=(d,)))

            t0 = time.time()
            hvp = self.compute_stochastic_vhp(hvp_fn, vec, params, batches)
            hvp = ravel_pytree(hvp)[0]
            if jnp.any(jnp.isnan(hvp)):
                print('NaN encountered!')
            else:
                hvps_norms.append(jla.norm(hvp) ** 2)

            t1 = time.time()
            dt = t1 - t0
            print(f'Overall time {dt * 1000:.2f}ms')
            print(f'Norm so far: {np.mean(hvps_norms):.5f}')

        if self.writer:
            estimate = np.mean(hvps_norms)
            print("Estimate:", estimate)
            results = {"fro_norm": estimate, 'iter_num': self.iter_num, 'num_fro_vecs': num_fro_vecs}
            with open(os.path.join(self.hvp_dir, f'fro_norm_info.pt'), 'wb') as f:
                pickle.dump(results, f)

    def compute_signal_to_noise_ratio(self, params, grad_fn, get_batch_fn):

        num_snr_batches = self.hessian_config['num_snr_batches']
        gradient_accumulation_steps = self.config['train']['batch']['gradient_accumulation_steps'] // self.world_size
        grad = ravel_pytree(jax.tree_map(jnp.zeros_like, params))[0]
        grad_sq = jnp.zeros_like(grad)

        for j in range(num_snr_batches):
            print('Batch:', j)

            batches = []
            for _ in range(gradient_accumulation_steps):
                batches.append(get_batch_fn('train'))

            grad_vec = self.compute_stochastic_gradient(grad_fn, params, batches)
            grad += grad_vec
            grad_sq += jnp.pow(grad_vec, 2.)

        grad /= num_snr_batches
        grad_sq /= num_snr_batches
        noise_sq = grad_sq - jnp.pow(grad, 2.)
        signal_sq = jnp.pow(grad, 2.)
        snr_vec = jnp.sqrt(signal_sq / noise_sq)
        l2_snr = jnp.sqrt(jnp.sum(signal_sq) / jnp.sum(noise_sq))
        avg_snr = snr_vec.mean()
        np.set_printoptions(precision=15)
        print('l2 snr:', l2_snr, 'avg snr:', avg_snr)
        results = {"l2_snr": l2_snr, 'avg_snr': avg_snr, 'snr_vec': snr_vec, 'num_snr_batches': num_snr_batches}
        with open(os.path.join(self.snr_dir, f'snr_info.np'), 'wb') as f:
            pickle.dump(results, f)

    def compute_grad_l1_norm_info(self, params, grad_fn, get_batch_fn):

        num_snr_batches = self.hessian_config['num_snr_batches']
        gradient_accumulation_steps = self.config['train']['batch']['gradient_accumulation_steps'] // self.world_size
        grad = ravel_pytree(jax.tree_map(jnp.zeros_like, params))[0]
        grad_sq = jnp.zeros_like(grad)

        t0 = time.time()
        for j in range(num_snr_batches):
            batches = []
            for _ in range(gradient_accumulation_steps):
                batches.append(get_batch_fn('train'))

            grad_vec = self.compute_stochastic_gradient(grad_fn, params, batches)
            grad += grad_vec
            grad_sq += jnp.pow(grad_vec, 2.)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"Batch {j}: time {dt * 1000:.2f}ms")

        grad /= num_snr_batches
        grad_sq /= num_snr_batches
        std = jnp.sqrt(grad_sq)
        grad_l1 = jla.norm(grad, ord=1)
        std_l1 = jla.norm(std, ord=1)
        np.set_printoptions(precision=15)
        print('Ratio:', grad_l1 / std_l1)
        results = {'grad': grad, 'std': std, 'ratio': grad_l1 / std_l1, 'num_snr_batches': num_snr_batches}
        with open(os.path.join(self.snr_dir, f'grad_std_info.np'), 'wb') as f:
            pickle.dump(results, f)

