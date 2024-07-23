import jsonlines
import pandas as pd
import os.path as osp
from functools import cache

import utils
from config import CANDIDATE_BUFFER_PATH, FEATURE_PATH, RXR_PRETRAIN_DIR

from .R2R_pretrain import R2R_MLMDataset


class RxR_MLMDataset(R2R_MLMDataset):
    def __init__(self, split, use_aug_data=False):
        self.dataset = load_data(split)
        if use_aug_data:
            dataset_aug = load_data('aug')
            self.dataset = pd.concat([self.dataset, dataset_aug], axis=0, copy=False)
        
        self.features = utils.load_features(FEATURE_PATH)
        self.candidate_buffer = utils.load_features(CANDIDATE_BUFFER_PATH)


RxR_ITMDataset = RxR_MLMDataset


@cache
def load_data(split):
    path = osp.join(RXR_PRETRAIN_DIR, f'RxRPretrain_{split}.jsonl')
    print(f'Loading from {path}')

    dataset = []
    with jsonlines.open(path) as data:
        for line in data:
            sample = dict()
            sample['scan'] = line['scan']
            sample['path'] = line['path']
            sample['path_id'] = line['path_id']
            sample['heading'] = line['heading']
            # sample['language'] = line['language']
            sample['instruction'] = line['instruction_ids'][:512]  # truncate
            sample['path_view_index'] = line['path_view_index']
            dataset.append(sample)
    return pd.DataFrame(dataset)
