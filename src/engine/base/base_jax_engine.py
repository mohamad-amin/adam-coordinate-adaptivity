import math
import os
import pickle

import jax
import wandb
import torch
import numpy as np
from jax import numpy as jnp
from contextlib import nullcontext
from transformers import AutoConfig
from torch.utils.tensorboard import SummaryWriter
from flax.training.common_utils import shard

from src.model.flax_model import FlaxGPT2LMHeadModel
from src.util.torch_to_flax import transfer_state_dict_to_params_and_rotations
from src.util.util import load_config, remove_unwanted_prefix_from_keys


class BaseJaxEngine(object):

    # Seed should be done
    def __init__(self, config_path, logger, load_dir, save_dir, tag):

        self.world_size = len(jax.devices())

        # Assign a logger
        self.logger = logger

        # Load configurations
        self.config = load_config(config_path)

        self.data_config = self.config['data']
        self.model_config = self.config['model']
        self.log_config = self.config['log']
        self.register_configs()

        # 'float64', 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        if self.model_config.get('dtype', 'float16') == 'float16':
            self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        else:
            self.dtype = self.model_config.get('dtype', 'float16')
        self.logger.info(f'Using dtype {self.dtype}')

        # note: float16 data type will automatically use a GradScaler
        self.jdtype = {'float64': jnp.float64, 'float32': jnp.float32, 'bfloat16': jnp.bfloat16, 'float16': jnp.float16}[self.dtype]

        # Load a summary writer
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(load_dir, 'checkpoints')

        self.id = self.prepare()

        if self.log_config['enable_wandb']:
            pass  # No wandb for release

        log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        del self.checkpoint

    def register_configs(self):
        pass

    def prepare(self):
        self._prepare_data(self.data_config)
        self._prepare_model(self.model_config)
        if self.checkpoint:
            id = self.checkpoint['wandb_id']
        else:
            id = wandb.util.generate_id()
        return id

    def _prepare_data(self, data_config):

        data_dir = data_config.get('dir', None)
        if data_dir is None:
            data_dir = os.path.join(os.environ['SLURM_TMPDIR'], 'myNanoGPT')
        data_dir = os.path.join(data_dir, data_config['name'])

        # poor man's data loader
        # preferably no os.environ here (for later)
        val_shard = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        train_bulk = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        # No sharding, only one process here
        train_shard = train_bulk

        if data_config.get('preload_data', True):
            print('Preloading data')
            data_train = jnp.asarray(train_shard.astype(jnp.int32))
            data_test = jnp.asarray(val_shard.astype(jnp.int32))
        else:
            print('Not preloading data')
            data_train = train_shard
            data_test = val_shard
        self.data = (data_train, data_test)

        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        self.meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.meta_vocab_size = meta['vocab_size']
            self.logger.info(f"found vocab_size = {self.meta_vocab_size} (inside {meta_path})")

    def _prepare_model(self, model_config):

        model_args = {k: model_config[k] for k in
                      set(list(model_config.keys())) - {'compile', 'init', 'rotate', 'rotations', 'dtype'}}

        if model_config['init'] == 'resume':
            self.logger.info(f"Resuming training from {self.load_dir}, save directory is set to {self.save_dir}.")
            # resume training from a checkpoint.
            init_iter = model_config.get('init_iter', -1)
            ckpt_path = os.path.join(self.checkpoint_dir, 'ckpt.pt' if init_iter == -1 else f'ckpt_{init_iter}.pt')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from config
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                try:
                    model_args[k] = checkpoint_model_args[k]
                except:
                    pass
            # create the model
            # Todo: update this, for now we're sticking to nanogpt
            config = AutoConfig.from_pretrained('openai-community/gpt2', local_files_only=True)
            config.update({
                "attn_pdrop": 0.,
                "embd_pdrop": 0.,
                "resid_pdrop": 0.,
                "summary_first_dropout": 0.,
                "vocab_size": 50304 if self.meta_vocab_size is None else self.meta_vocab_size,
                "activation_function": "gelu",
                "n_embd": model_config['n_embd'],
                "n_head": model_config['n_head'],
                "n_layer": model_config['n_layer'],
                "n_positions": self.config['train']['batch']['block'],
                "scale_attn_by_inverse_layer_idx": False,  # Todo
            })
            model = FlaxGPT2LMHeadModel._from_config(config, dtype=self.jdtype)
            print(f'Built model with dtype {self.jdtype}')
            state_dict = remove_unwanted_prefix_from_keys(checkpoint['model'])
            params, rotations = transfer_state_dict_to_params_and_rotations(state_dict, model.params)
            model.params = jax.tree_util.tree_map(
                lambda p: p if p.dtype == self.jdtype else p.astype(self.jdtype),
            model.params)
            self.checkpoint_rotations = rotations  # Will be an empty dict if the checkpoint is not rotated
            self.model = model
            self.checkpoint = checkpoint
            self.model_args = model_args
            # assert model_config.get('rotate', False) is False, "Original GPT can't be rotated."
        else:
            raise NotImplementedError('Jax is solely used for evaluations!')

    def get_batch(self, split, batch_size, block_size, do_shard=True):
        if split == 'train':
            data = self.data[0]
        else:
            data = self.data[1]
        ix = np.random.randint(0, len(data) - block_size, (batch_size,))
        if self.data_config.get('preload_data', True):
            x = jnp.stack([data[i:i + block_size] for i in ix])
            y = jnp.stack([data[i + 1:i + 1 + block_size] for i in ix])
        else:
            x = jnp.stack([data[i:i + block_size].astype(jnp.int32) for i in ix])
            y = jnp.stack([data[i+1:i + 1 + block_size].astype(jnp.int32) for i in ix])
        return {
            'input_ids': shard(x) if do_shard else x,
            'labels': shard(y),
            'attention_mask': shard(jnp.ones_like(x)) if do_shard else jnp.ones_like(x)
        }

    def run(self):
        pass
