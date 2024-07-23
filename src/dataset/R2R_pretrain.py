import math
import torch
import random
import jsonlines
import numpy as np
import pandas as pd
import os.path as osp
import networkx as nx
from functools import cache

import utils
from config import CANDIDATE_BUFFER_PATH, FEATURE_PATH, R2R_PRETRAIN_DIR

from .graph import connectivity
from .R2R import get_teacher_selection, get_candidate_features, get_pano_feature


class R2R_MLMDataset:
    def __init__(self, split, use_aug_data=False):
        self.dataset = load_data(split)
        if use_aug_data:
            dataset_aug = load_data('aug')
            self.dataset = pd.concat([self.dataset, dataset_aug], axis=0, copy=False)

        self.features = utils.load_features(FEATURE_PATH)
        self.candidate_buffer = utils.load_features(CANDIDATE_BUFFER_PATH)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            path_id, text_id, history_panorama, history_length, candidates
        """
        sample = self.dataset.iloc[idx]

        # path_id
        path_id = sample['path_id']

        # text
        text_id = sample['instruction']  # list[int]
        text_id = torch.tensor(text_id)

        # history and view_indexes
        scan = sample['scan']
        path = sample['path']
        view_index_list = sample['path_view_index']
        history_panorama = []
        history_candidates = []
        history_teacher_selection = []
        for i, viewpoint_id in enumerate(path):
            view_index = view_index_list[i]
            heading = math.radians(30) * (view_index % 12)
            elevation = math.radians(30) * (view_index // 12 - 1)

            long_id = '%s_%s' % (scan, viewpoint_id)
            pano_feature = get_pano_feature(heading, elevation, self.features[long_id])
            history_panorama.append(pano_feature)

            candidates = self.candidate_buffer[long_id].copy()
            candidates['vision_feature'] = [self.features[long_id][view_index] for view_index in candidates['view_index']]
            candidate_feature = get_candidate_features(heading, elevation, candidates)  # (candidate_num, 768)
            candidate_feature = torch.from_numpy(candidate_feature)
            history_candidates.append(candidate_feature)

            next_viewpoint_id = path[i+1] if i + 1 < len(path) else None
            teacher_selection = get_teacher_selection(candidates, next_viewpoint_id)
            history_teacher_selection.append(teacher_selection)
        history_panorama = np.stack(history_panorama)
        history_panorama = torch.from_numpy(history_panorama)
        history_length = len(history_panorama)
        history_teacher_selection = torch.tensor(history_teacher_selection)

        # history_candidates is list
        return path_id, text_id, history_panorama, history_length, history_candidates, history_teacher_selection


R2R_ITMDataset = R2R_MLMDataset


@cache
def load_data(split):
    path = osp.join(R2R_PRETRAIN_DIR, f'R2RPretrain_{split}.jsonl')
    print(f'Loading from {path}')

    dataset = []
    with jsonlines.open(path) as data:
        for line in data:
            instructions = line['instructions_ids']
            for instruction in instructions:
                sample = dict()
                sample['scan'] = line['scan']
                sample['path'] = line['path']
                sample['path_id'] = line['path_id']
                sample['heading'] = line['heading']
                sample['instruction'] = instruction
                sample['path_view_index'] = line['path_view_index']
                dataset.append(sample)
    return pd.DataFrame(dataset)
