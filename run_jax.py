import argparse
import os
import shutil

import torch
import yaml


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
    parser.add_argument('--config_path', default='', help="Path to a config")
    parser.add_argument('--save_dir', default='', help='Path to dir to save checkpoints and logs')
    parser.add_argument('--load_dir', default='', help='Path to dir to load checkpoints and logs')
    parser.add_argument('--specific_tag', default=None, help='specific tag for this run used by wandb')
    args = parser.parse_args()

    if args.load_dir is None:
        args.load_dir = args.save_dir

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    from src.util.util import load_log
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))

    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    logging_name = ''
    logging_name += args.save_dir
    logger = load_log(logging_name)

    set_seeds(args.config_path, 0)

    import jax
    logger.info(f'World size is {jax.device_count()}.')

    if 'hessian' in config:
        hessian_config = config['hessian']
        from src.engine.hessian.jax_hessian_eval_engine import HessianEvalEngine
        HessianEvalEngine(args.config_path, logger, args.load_dir, args.save_dir, args.specific_tag).run()
    else:
        exit('No other mode than evaluation, for now...')
