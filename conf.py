# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = 'Standard'

# Choice of (source, norm, tent)
# - source: baseline without adaptation
# - norm: test-time normalization
# - tent: test-time entropy minimization (ours)
_C.MODEL.ADAPTATION = 'ours'
_C.MODEL.NUM_CLASSES = 10

# By default tent is online, with updates persisting across batches.
# To make adaptation episodic, and reset the model for each batch, choose True.
_C.MODEL.EPISODIC = False


# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# COTTA
_C.OPTIM.MT = 0.99
_C.OPTIM.RST = 0.01
_C.OPTIM.AP = 0.92

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate (10000 for all samples in CIFAR-10)
_C.CORRUPTION.NUM_EX = 10000


# ------------------------------- Federated Learning options ------------------------------ #
_C.FED = CfgNode()
# Number of clients in the federated learning setup
_C.FED.NUM_CLIENTS =  20             # Number of clients
# Number of samples in each batch
_C.FED.BATCH_SIZE = 10             # Batch size for each client
# Number of steps for each client
_C.FED.NUM_STEPS = 750              # Number of steps for each client
_C.FED.SPATIAL_H = 0.2
_C.FED.TEMPORAL_H = 0.02
_C.FED.IID = True
_C.FED.AGG_FREQ = 1

# ------------------------------- Misc options ------------------------------ #
_C.MISC = CfgNode()
_C.MISC.RNG_SEED = 2
_C.MISC.SAVE_DIR = "./output"
_C.MISC.DATA_DIR  =  "./data"
_C.MISC.CKPT_DIR  = "./ckpt"
_C.MISC.LOG_DEST = 'log.txt'
_C.MISC.LOG_TIME = ''
_C.MISC.ADAPT_ALL = True
# Aggregation based on similarity of weigths or ema_probs
_C.MISC.SIMILARITY = 'weights'

_C.MISC.MOMENTUM_TEACHER = 0.99
_C.MISC.MOMENTUM_PROBS = 0.9
_C.MISC.TEMP = 0.01
_C.MISC.EMA_PROBS_TEMP = 1

# ------------------------------- CUDNN options ------------------------------ #

_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True


# ------------------------------- BN options ------------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1



def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, default='configs/cifar10.yaml',
                        help="Config file location")
    parser.add_argument("--opts", nargs='+', help="Modify config options using the format KEY VALUE")

    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}_{}.txt'.format(cfg.MODEL.ADAPTATION, current_time))
    iid_text = 'IID' if cfg.MISC.IID else 'NIID'

    g_pathmgr.mkdirs(os.path.join(cfg.MISC.SAVE_DIR, iid_text))
    cfg.MISC.LOG_TIME, cfg.MISC.LOG_DEST = current_time, log_dest
    cfg.MISC.NUM_STEPS = cfg.CORRUPTION.NUM_EX  * len(cfg.CORRUPTION.TYPE)// (cfg.FED.BATCH_SIZE * cfg.FED.NUM_CLIENTS)
    
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.MISC.SAVE_DIR, iid_text, cfg.MISC.LOG_DEST)),
            logging.StreamHandler()
        ])

    np.random.seed(cfg.MISC.RNG_SEED)
    torch.manual_seed(cfg.MISC.RNG_SEED)
    random.seed(cfg.MISC.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)