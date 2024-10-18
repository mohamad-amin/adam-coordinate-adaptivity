import math
import os
import pickle

import wandb
import torch
import numpy as np
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.model import GPTConfig, GPT
from src.util.util import load_config, remove_unwanted_prefix_from_keys

import torch._dynamo
torch._dynamo.config.suppress_errors = True


class BaseEngine(object):

    # Seed should be done
    def __init__(self, config_path, logger, save_dir, master_process, local_rank, rank, world_size, tag):

        self.ddp = local_rank != -1
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        # Assign a logger
        self.logger = logger

        # Load configurations
        self.config = load_config(config_path)

        self.data_config = self.config['data']
        self.model_config = self.config['model']
        self.log_config = self.config['log']
        self.register_configs()

        # Determine which device to use
        if torch.cuda.is_available():
            self.device_type = 'cuda'
            self.device = f'cuda:{local_rank}' if local_rank != -1 else 'cuda'
        else:
            self.device_type = 'cpu'
            self.device = 'cpu'
            self.logger.warn('Cuda is not available! Running on CPU.')
        # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        if self.model_config.get('dtype', 'float16') == 'float16':
            self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        else:
            self.dtype = 'float32'
        if master_process:
            self.logger.info(f'Using dtype {self.dtype}')
        # note: float16 data type will automatically use a GradScaler
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' \
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

        # Load a summary writer
        self.save_dir = save_dir
        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints')

        self.id = self.prepare()
        if master_process:

            if self.log_config['enable_wandb']:
                pass  # No wandb for release

            log_dir = os.path.join(self.save_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        else:
            self.writer = None

        del self.checkpoint
        if self.ddp:
            self.model = DDP(self.model, device_ids=[local_rank])

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
        if data_config.get('shard_data', False) and self.ddp:
            # take a slice, train on that slice
            train_count = len(train_bulk)
            shard_size = math.ceil(train_count / self.world_size)
            train_shard = train_bulk[int(self.rank * shard_size):int((self.rank + 1) * shard_size)]
        else:
            train_shard = train_bulk

        if data_config.get('preload_data', True):
            if self.rank == 0:
                print('Preloading data')
            data_train = torch.from_numpy(train_shard.astype(np.int32))
            data_test = torch.from_numpy(val_shard.astype(np.int32))
        else:
            if self.rank == 0:
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

        if model_config['init'] == 'scratch':
            # init a new model from scratch
            if self.rank == 0:
                self.logger.info("Initializing a new model from scratch")
            # determine the vocab size we'll use for from-scratch training
            if self.meta_vocab_size is None:
                self.logger.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
            model_args['vocab_size'] = self.meta_vocab_size if self.meta_vocab_size is not None else 50304
            gptconf = GPTConfig(**model_args)
            self.model = GPT(gptconf)
            self.checkpoint = None
            self.model_args = model_args
            self.model_args['block_size'] = gptconf.block_size
        elif model_config['init'] == 'resume':
            if self.rank == 0:
                self.logger.info(f"Resuming training from {self.save_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(self.checkpoint_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from config
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                try:
                    model_args[k] = checkpoint_model_args[k]
                except:
                    pass
            # create the model
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = remove_unwanted_prefix_from_keys(checkpoint['model'])
            model.load_state_dict(state_dict, strict=False)
            self.model = model
            self.checkpoint = checkpoint
            self.model_args = model_args
        elif model_config['init'].startswith('gpt2'):
            assert model_config.get('rotate', False) is False, "Original GPT can't be rotated."
            self.logger.info(f"Initializing from OpenAI GPT-2 weights: {model_config['init']}")
            # initialize from OpenAI GPT-2 weights
            override_args = dict(dropout=model_config['dropout'])
            self.model = GPT.from_pretrained(model_config['init'], override_args)
            self.checkpoint = None
            # read off the created config params, so we can store them into checkpoint correctly
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = getattr(self.model.config, k)
            self.model_args = model_args

        # crop down the model block size if desired, using model surgery
        # Todo: this is ugly, requires train_config in building the model
        block_size = self.config['train']['batch']['block']
        if block_size < self.model.config.block_size:
            self.model.crop_block_size(block_size)
            self.model_args['block_size'] = block_size  # so that checkpoint will have the right value

        self.model.to(self.device)

        # compile the model
        if model_config['compile']:
            self.logger.info("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)  # requires PyTorch 2.0

    def get_batch(self, split, batch_size, block_size):
        if split == 'train':
            data = self.data[0]
        else:
            data = self.data[1]
        ix = torch.randint(len(data) - block_size, (batch_size,))
        if self.data_config.get('preload_data', True):
            x = torch.stack([data[i:i + block_size] for i in ix])
            y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
        else:
            x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().long().to(self.device, non_blocking=True), \
                   y.pin_memory().long().to(self.device, non_blocking=True)
        else:
            x, y = x.long().to(self.device), y.long().to(self.device)
        return x, y

    def wrap_ddp(self, model):
        # if self.ddp and not isinstance(model, DDP):
        if self.ddp:
            return DDP(model, device_ids=[self.local_rank])
        else:
            return model

    def unwrap_ddp(self, model):
        # if self.ddp and isinstance(model, DDP):
        if self.ddp:
            return model.module
        else:
            return model

    def run(self):
        pass
