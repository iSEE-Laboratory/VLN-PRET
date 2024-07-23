from .logger import Logger
from .pbar import PrintProgress

from .loss import cross_entropy_loss
from .lr_scheduler import LRScheduler
from .dtw import load_graph, ndtw_initialize
from .mask import get_causal_mask, length_to_mask, merge_mask


import functools
import traceback
def print_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print('\nRANK:', args[0])
            traceback.print_exception(type(e), e, e.__traceback__)
        exit()
    return wrapper


import os
import torch
import random
import numpy as np
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量


import socket
def get_available_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    return port


import torch.distributed as dist
class EndlessDataLoader:
    def __init__(self, dataloader):
        self.epoch = 0
        self.dataloader = dataloader
    
    def __iter__(self):
        while True:
            for batch in self.dataloader:
                yield batch
            if dist.is_initialized():
                self.dataloader.sampler.set_epoch(self.epoch)
            self.epoch += 1


import pickle
from functools import cache
@cache  # avoid reload feature
def load_features(feature_path):
    print('Loading %s' % feature_path)
    with open(feature_path, 'rb') as f:
        features = pickle.load(f)
    return features


import cv2
import numpy as np
def to_numpy(img):
    bgr = np.array(img, copy=False)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def get_angle_feature(heading, elevation, tile=True):
    """
    Args:
        heading: (N,) array
        elevation: (N,) array
    Returns:
        (N, 128) array if tile else (N, 4)
    """
    feature = np.stack([np.sin(heading), np.cos(heading), np.sin(elevation), np.cos(elevation)], axis=1)
    if tile:
        feature = np.tile(feature, (1, 32))
    return feature


def MLM(input_ids, attention_mask, mask_prob=0.15, sep_token_id=102, mask_token_id=103, vocab_size=30522):
    """masked language modeling.
    Args:
        input_ids: tensor
        attention_mask: tensor
        mask_prob: float
        sep_token_id: int
        mask_token_id: int
        vocab_size: int
    Returns:
        input_ids: masked input_ids
        masked_mask: a mask indicate wich elements are paded.
        masked_ids: origin ids of masked elements.
    """
    device = input_ids.device
    rand = torch.rand_like(input_ids, dtype=torch.float32, device=device)

    masked_mask = rand < mask_prob
    masked_mask = masked_mask * attention_mask.bool()  # remove mask on PAD
    masked_mask[:, 0] = False  # remove mask on CLS
    masked_mask[input_ids == sep_token_id] = False  # remove mask on SEP

    masked_ids = input_ids[masked_mask]

    # 80% replace MASK, 10% random replace, 10% not change
    random_replace_mask = (0.8 * mask_prob <= rand) & (rand < 0.9 * mask_prob) & masked_mask
    not_change_mask = (0.9 * mask_prob <= rand) & masked_mask

    input_ids = input_ids.clone()
    input_ids[masked_mask & ~not_change_mask] = mask_token_id
    input_ids[random_replace_mask] = torch.randint_like(
        input_ids,
        high=vocab_size, device=device)[random_replace_mask]
    
    return input_ids, masked_mask, masked_ids
