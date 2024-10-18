import argparse
import os
import shutil
from datetime import timedelta

import torch
import yaml

from torch.distributed import init_process_group, destroy_process_group


def set_seeds(config_path, seed_offset):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    seed = config['data'].get('seed', 1313) + seed_offset

    # Fixing the seeds for data
    import numpy
    import random
    numpy.random.seed(seed)
    random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',
                        help="Path to a config")
    parser.add_argument('--save_dir', default='',
                        help='Path to dir to save checkpoints and logs')
    parser.add_argument('--specific_tag', default=None, help='specific tag for this run used by wandb')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    from src.util.util import load_log
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))

    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?

    if ddp:
        init_process_group(backend='nccl', timeout=timedelta(seconds=10800))
        ddp_rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        logging_name = str(ddp_rank) + ':'
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        local_rank = -1
        ddp_rank = -1
        world_size = 1
        logging_name = ''

    logging_name += args.save_dir
    logger = load_log(logging_name)

    set_seeds(args.config_path, seed_offset)

    if master_process:
        logger.info(f'World size is {world_size}.')

    if 'hessian' in config:
        from src.engine.hessian.hessian_eval_engine import HessianEvalEngine
        HessianEvalEngine(args.config_path,
           logger, args.save_dir, master_process, local_rank, ddp_rank, world_size, args.specific_tag).run()
    elif 'train' in config:
        if config['model'].get('rotate', False):
            from src.engine.rotated_train_engine import RotatedTrainEngine
            engine = RotatedTrainEngine
        else:
            from src.engine.train_engine import TrainEngine
            engine = TrainEngine
        engine(args.config_path,
               logger, args.save_dir, master_process, local_rank, ddp_rank, world_size, args.specific_tag).run()
    else:
        exit('No other mode than training, for now...')

    if ddp:
        destroy_process_group()
