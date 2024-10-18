import copy
from functools import partial
import torch.distributed as dist

from src.engine.train_engine import TrainEngine
from src.util.rotation.rotations import *
from src.util.util import remove_unwanted_prefix_from_keys

ROTATION_DICT = {
    'weight': WeightRotation,
    'weightrp': WeightRandPermRotation,
    'block_mix': BlockMixingRotation,
    'layer_norm_mix': LayerNormMixingRotation,
}


class RotatedTrainEngine(TrainEngine):

    def prepare(self):
        super().prepare()
        self._prepare_rotated_model()
        if self.train_config.get('manual_rotate_investigation', False):
            import IPython; IPython.embed()

    def _prepare_rotated_model(self):

        self.rotated_model = self.wrap_ddp(copy.deepcopy(self.model))
        raw_model = self.get_raw_model()
        rotator_states = None

        if self.checkpoint:

            own_state_keys = remove_unwanted_prefix_from_keys(raw_model.state_dict()).keys()
            self.checkpoint['model'] = remove_unwanted_prefix_from_keys(self.checkpoint['model'])
            checkpoint_state_dict_keys = self.checkpoint['model'].keys()

            for name in list(checkpoint_state_dict_keys):
                param = self.checkpoint['model'][name]
                if name not in own_state_keys:  # orthogonal rotation buffers
                    buffer = torch.tensor(param, requires_grad=False, device=self.device)
                    module = raw_model
                    for module_name in name.split('.')[:-1]:
                        module = getattr(module, module_name)
                    module.register_buffer(name.split('.')[-1], buffer)
                del self.checkpoint['model'][name]
                torch.cuda.empty_cache()

            keys = list(self.checkpoint['optimizer']['state'].keys())
            for key in keys:
                del self.checkpoint['optimizer']['state'][key]

            rotator_states = self.checkpoint['rotations']
            torch.cuda.empty_cache()

        def make_rotation(name):
            if name.startswith('weightrp'):
                start_idx = name.find('_')
                k = int(name[start_idx + 1:start_idx+2])
                name = name[start_idx + 2:]
                start_idx = name.find('__')
                args_string = name[start_idx + 2:]
                weight_types = [arg.strip() for arg in args_string.split('.')]
                return partial(WeightRandPermRotation, weight_types=weight_types, k=k)
            elif name.startswith('weight'):
                start_idx = name.find('__')
                args_string = name[start_idx + 2:]
                weight_types = [arg.strip() for arg in args_string.split('.')]
                print(weight_types)
                return partial(WeightRotation, weight_types=weight_types)
            elif name.startswith('individualrp'):
                start_idx = name.find('_')
                k = int(name[start_idx + 1:])
                return partial(IndividualRandPermRotation, k=k)
            elif name.startswith('randperm'):
                start_idx = name.find('_')
                k = int(name[start_idx + 1:])
                return partial(RandomPermuteRotation, k=k)
            else:
                return ROTATION_DICT[name]

        self.rotator = Rotator.Compose(raw_model, [
            make_rotation(rotation_name) for rotation_name in self.model_config.get('rotations')
        ], rotator_states)

        # Syncing rotations
        if self.world_size > 1:
            for buffer_name, buffer in raw_model.named_buffers():
                dist.broadcast(buffer.data, src=0)
            for rotation in self.rotator.rotations:
                if hasattr(rotation, 'state') and rotation.state is not None:
                    dist.broadcast_object_list(rotation.state, src=0)

        self.logger.info(f'Rotations are: {self.model_config.get("rotations")}')

        if self.checkpoint:
            self.rotator.backward_weight()
            self.model.load_state_dict(raw_model.state_dict(), strict=False)

    def get_raw_model(self):
        return self.unwrap_ddp(self.rotated_model)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def _estimate_loss(self, get_batch):

        self.rotated_model.load_state_dict(self.model.state_dict(), strict=False)
        self.rotated_model.eval()
        self.rotator.forward()
        out = {}

        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_config['iters'])
            for k in range(self.eval_config['iters']):
                X, Y = get_batch(split)
                with self.ctx:
                    logits, loss = self.rotated_model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.rotated_model.train()
        return out

    # Keep track of parameters
    @torch.no_grad()
    def _measure_params(self):

        self.rotated_model.load_state_dict(self.model.state_dict(), strict=False)
        self.rotated_model.eval()
        self.rotator.forward()
        m = self.unwrap_ddp(self.rotated_model)
        out = {}

        l1_norms = []
        linf_norms = []
        for n, p in m.named_parameters():
            out['L2_' + n] = torch.norm(p)
            out['L1_' + n] = torch.norm(p, 1)
            out['Linf_' + n] = torch.abs(p).max()
            l1_norms.append(out['L1_' + n])
            linf_norms.append(out['Linf_' + n])

        out['Linf'] = torch.max(torch.tensor(linf_norms))
        out['L1'] = torch.max(torch.tensor(l1_norms))
        return out

    def forward_backward(self, gradient_accumulation_steps, get_batch, X, Y, scaler):

        self.rotated_model.load_state_dict(self.model.state_dict(), strict=False)
        self.rotator.forward()

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if self.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                self.rotated_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with self.ctx:
                logits, loss = self.rotated_model(X, Y)
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        self.rotator.backward()

        parameters = self.unwrap_ddp(self.model).parameters()
        rotated_parameters = self.unwrap_ddp(self.rotated_model).parameters()
        for param, rotated_param in zip(parameters, rotated_parameters):
            if rotated_param.grad is not None:
                param.grad = rotated_param.grad.clone()
                rotated_param.grad = None
        self.rotated_model.zero_grad(set_to_none=True)

        return X, Y, loss

    def make_checkpoint(self):
        return {
            'rotations': self.rotator.dump_state(),
            'model': self.get_raw_model().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.model_args,
            'config': self.config,
            'wandb_id': self.id
        }