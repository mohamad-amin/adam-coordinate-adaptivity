import pickle
from pathlib import Path
from typing import Union
from tqdm import trange
import hashlib
import json
import torch
import logging
import os
import yaml
import numpy as np
from colorlog import ColoredFormatter

suffix = ".pytree"


def flatten_tree(p, label=None):
  if isinstance(p, dict):
    for k, v in p.items():
      yield from flatten_tree(v, k if label is None else f"{label}.{k}")
  else:
    yield label, p


def write_stats(d, writer, step):
    for top_k in d.keys():
        for k in d[top_k].keys():
            val = d[top_k][k]
            val = val.item() if hasattr(val, 'item') else val
            writer.add_scalar(top_k + '/' + k, val, global_step=step)


def UID(config):
    dhash = hashlib.md5()
    encoded = json.dumps(config, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def tree_save(data, path: Union[str, Path], overwrite: bool = False):
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f"File {path} already exists.")
    with open(path, "wb") as file:
        pickle.dump(data, file)


def tree_load(path: Union[str, Path]):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix != suffix:
        raise ValueError(f"Not a {suffix} file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


# Logging
# =======

def load_log(name):
    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOV':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')
    logging.Logger.infov = _infov
    return log


# General utils
# =============

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


# Path utils
# ==========

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
    return path


# Data
# ====

def to_device(dict, device):
    for key in dict.keys():
        if isinstance(dict[key], torch.Tensor):
            dict[key] = dict[key].to(device)
    return dict


# Divisors of a number
# ====================

def get_sorted_divisors(num):
    candidates = np.arange(1, num+1)
    return candidates[np.mod(num, candidates) == 0]


# Torch state dict
# ====================
def remove_unwanted_prefix_from_keys(state_dict, unwanted_prefix='_orig_mod.'):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict
