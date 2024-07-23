"""
Room-to-Room dataset.
"""

import json
import math
import torch
import pickle
import numpy as np
import pandas as pd
import os.path as osp
import torch.distributed as dist
from torch.utils.data import RandomSampler, BatchSampler, DistributedSampler

import utils
from config import CANDIDATE_BUFFER_PATH, FEATURE_PATH, R2R_DIR

from .graph import connectivity
from .mattersim import ACTION, MatterSimEnvBatch


class R2R:
    def __init__(self, split, render=False, img_size=(640, 480), vfov=60):
        self.vfov = vfov
        self.render = render
        self.img_size = img_size

        self.split = split
        self.dataset = load_R2R_dataset(split)  # pd.DataFrame

        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        s1 = f'batch size: {self.batch_size}\n'
        s2 = f'dataset size: {len(self.dataset)}\n'
        return s1 + s2 + str(self.dataset)

    def __getitem__(self, ids):
        """
        ids: list[int]
        """
        batch = self.dataset.loc[ids]
        if isinstance(ids, list):
            batch.reset_index(inplace=True)
        env = R2REnvBatch(batch, render=self.render, img_size=self.img_size, vfov=self.vfov)
        return batch, env

    def get_batches(self, batch_size):
        if dist.is_initialized():
            sampler = DistributedSampler(self.dataset, shuffle=True, drop_last=False)
            sampler.set_epoch(self.epoch)
        else:
            # use epoch num as seed for reproduction
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            sampler = RandomSampler(self.dataset, generator=generator)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)

        for batch_ids in batch_sampler:
            batch = self.dataset.loc[batch_ids].reset_index()
            env = R2REnvBatch(batch, render=self.render, img_size=self.img_size, vfov=self.vfov)
            yield batch, env
        self.epoch += 1


class R2REnvBatch:
    env_buffer = dict()

    def __init__(self, batch, render=False, img_size=(640, 480), vfov=60):
        self.batch = batch  # pandas DataFrame
        self.render = render
        self.batch_size = len(batch)

        if self.batch_size not in R2REnvBatch.env_buffer:
            self.env = MatterSimEnvBatch(self.batch_size, render=render, img_size=img_size, vfov=vfov)
            R2REnvBatch.env_buffer[self.batch_size] = self.env
        else:
            self.env = R2REnvBatch.env_buffer[self.batch_size]
        
        self.feature = utils.load_features(FEATURE_PATH)
        self.candidate_buffer = utils.load_features(CANDIDATE_BUFFER_PATH)
    
    # @profile
    def _get_obs(self, scan_id, viewpoint_id, view_index, rgb, path):
        """Get observation of a sample.
        """
        heading = math.radians(30) * (view_index % 12)
        elevation = math.radians(30) * (view_index // 12 - 1)

        goal_viewpoint_id = path[-1]
        distance_to_goal = connectivity.get_distance(scan_id, viewpoint_id, goal_viewpoint_id)

        xyz = connectivity.get_graph(scan_id).nodes[viewpoint_id]['pos3d']

        long_id = '%s_%s' % (scan_id, viewpoint_id)
        candidates = self.candidate_buffer[long_id].copy()
        candidates['vision_feature'] = [self.feature[long_id][view_index] for view_index in candidates['view_index']]
        candidate_feature = get_candidate_features(heading, elevation, candidates)

        panoramic_feature = get_pano_feature(heading, elevation, self.feature[long_id])

        if viewpoint_id in path:
            # NOTE: This is used in teacher forcing
            # for RxR dataset, simply take shortest path as target maybe suboptimal
            # if in gt path, then following the gt path rather than shortest path
            i = path.index(viewpoint_id)
            next_viewpoint_id = path[i + 1] if i + 1 < len(path) else None
        else:
            # NOTE: This is actually not used.
            # simply use shortest path as gt when deviating from gt path may be suboptimal
            # find a path to the goal that matches best with gt path should be better.
            shortest_path = connectivity.get_path(scan_id, viewpoint_id, goal_viewpoint_id)
            next_viewpoint_id = shortest_path[1] if len(shortest_path) > 1 else None
        teacher_selection = get_teacher_selection(candidates, next_viewpoint_id)

        observation = {
            'scan_id': scan_id,
            'viewpoint_id': viewpoint_id,  # str
            'xyz': xyz,
            'heading': heading,
            'elevation': elevation,
            'view_index': view_index,  # int
            'rgb': rgb,

            'distance_to_goal': distance_to_goal,
            'candidates': candidates,
            'candidate_feature': candidate_feature,
            'panoramic_feature': panoramic_feature,
            'next_viewpoint_id': next_viewpoint_id,
            'teacher_selection': teacher_selection,
        }
        return observation

    def _get_observations(self):
        observations = []
        states = self.env.get_states()
        for i, state in enumerate(states):
            scan_id = state.scanId
            viewpoint_id = state.location.viewpointId
            view_index = state.viewIndex
            rgb = utils.to_numpy(state.rgb) if self.render else None

            observation = self._get_obs(scan_id, viewpoint_id, view_index, rgb, self.batch.at[i, 'path'])
            observations.append(observation)
        return pd.DataFrame(observations, dtype=object, copy=False)

    # @profile
    def reset(self):
        scan_ids = self.batch['scan'].tolist()
        viewpoint_ids = [path[0] for path in self.batch['path']]
        headings = self.batch['heading'].tolist()
        self.env.new_episodes(scan_ids, viewpoint_ids, headings)  # 20%
        observations = self._get_observations()  # 80%
        return observations

    def step(self, actions, return_observation=True):
        self.env.step(actions)
        if return_observation:
            return self._get_observations()

    def step_viewpoint(self, observations, selections):
        """Move between viewpoint according to the navigation graph, not the simulator because it is slow.
        Returns:
            observation.
        """
        observations = observations.copy()
        for i, selection in enumerate(selections):
            if selection <= 0:  # STOP
                continue

            scan_id = observations.at[i, 'scan_id']

            candidates = observations.at[i, 'candidates']
            view_index = candidates.at[selection-1, 'view_index']
            viewpoint_id = candidates.at[selection-1, 'viewpoint_id']

            # assign a dict to a DataFrame line is slow.
            # observations.iloc[i] = self._get_obs(scan_id, viewpoint_id, view_index, None, path = self.batch.at[i, 'path'])
            line = self._get_obs(scan_id, viewpoint_id, view_index, None, self.batch.at[i, 'path'])
            line = list(line.values())
            observations.values[i] = line

        if self.render:
            self.env.new_episodes(
                observations['scan_id'].tolist(), observations['viewpoint_id'].tolist(),
                observations['heading'].tolist(), observations['elevation'].tolist())
            states = self.env.get_states()
            observations['rgb'] = [utils.to_numpy(state.rgb) for state in states]
        return observations

    # @profile
    def step_target(self, observations, batch_path):
        """Move to target viewpoint directly.
        Returns:
            observation.
        """
        observations = observations.copy()
        for i, path in enumerate(batch_path):
            if len(path) == 1:  # STOP
                continue

            scan_id = observations.at[i, 'scan_id']
            long_id = '%s_%s' % (scan_id, path[-2])
            candidates = self.candidate_buffer[long_id]

            selection = candidates['viewpoint_id'].tolist().index(path[-1])

            view_index = candidates.at[selection, 'view_index']
            viewpoint_id = candidates.at[selection, 'viewpoint_id']

            # assign a dict to a DataFrame line, this is slow.
            # observations.iloc[i] = self._get_obs(scan_id, viewpoint_id, view_index, None, self.batch.at[i, 'path'])
            line = self._get_obs(scan_id, viewpoint_id, view_index, None, self.batch.at[i, 'path'])
            line = list(line.values())
            observations.values[i] = line

        if self.render:
            self.env.new_episodes(
                observations['scan_id'].tolist(), observations['viewpoint_id'].tolist(),
                observations['heading'].tolist(), observations['elevation'].tolist())
            states = self.env.get_states()
            observations['rgb'] = [utils.to_numpy(state.rgb) for state in states]
        return observations


def load_R2R_dataset(split):
    cache_dir = osp.join(R2R_DIR, 'encoded_cache_dir')
    if osp.exists(cache_dir):
        if split == 'aug':
            path = osp.join(cache_dir, f'R2R_prevalent_aug.pkl')
        elif split == 'val_train':
            path = osp.join(cache_dir, f'R2R_val_train_seen.pkl')
        else:
            path = osp.join(cache_dir, f'R2R_{split}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        if split == 'aug':
            path = osp.join(R2R_DIR, f'R2R_prevalent_aug.json')
        elif split == 'val_train':
            path = osp.join(R2R_DIR, f'R2R_val_train_seen.json')
        else:
            path = osp.join(R2R_DIR, f'R2R_{split}.json')
        with open(path) as f:
            data = json.load(f)

    dataset = []
    for item in data:
        # each item corresponds to 1 path
        # each path corresponds to 3~4 instructions
        for i, instruction in enumerate(item['instructions']):
            sample = dict()  # a sample in dataset
            sample['scan'] = item['scan']
            sample['path'] = item['path']
            sample['path_id'] = item['path_id']
            sample['heading'] = item['heading']
            sample['instruction'] = instruction  # it may be str or list[int], str is not tokenized, list[int] is tokenized
            sample['instruction_id'] = '%s_%d' % (item['path_id'], i)  # for submitting test result 
            if 'tokens' in item:
                sample['tokens'] = item['tokens'][i]
            dataset.append(sample)
    return pd.DataFrame(dataset)


def get_candidate_features(heading, elevation, candidates):
    """
    Args:
        candidates: DataFrame
    Returns:
        a tensor of shape (N, C)
    """
    rel_headings = candidates['heading'].to_numpy(dtype=np.float32) - heading
    rel_elevations = candidates['elevation'].to_numpy(dtype=np.float32) - elevation
    angle_features = utils.get_angle_feature(rel_headings, rel_elevations)
    vision_features = np.stack(candidates['vision_feature'])
    candidate_features = np.concatenate([angle_features, vision_features], axis=1)
    return candidate_features


def get_pano_feature(heading, elevation, panoramic_feature):
    view_index = np.arange(0, 36, dtype=np.float32)
    rel_headings = (view_index % 12) * math.radians(30) - heading
    rel_elevations = (view_index // 12 - 1) * math.radians(30) - elevation
    angle_features = utils.get_angle_feature(rel_headings, rel_elevations)
    panoramic_feature = np.concatenate([angle_features, panoramic_feature], axis=1)
    return panoramic_feature


def get_teacher_selection(candidates, next_viewpoint_id):
    if next_viewpoint_id is None:
        action = 0  # STOP
    else:
        # where = (candidates['viewpoint_id'] == next_viewpoint_id)
        # action = candidates.index[where][0]
        action = candidates['viewpoint_id'].tolist().index(next_viewpoint_id)
        action += 1  # + 1 for action STOP
    return action


def selection_to_actions(selection, view_index, candidates):
    """
    Args:
        selection: int
        view_index: integer range from [0, 35]
        candidates: DataFrame, each row is a candidate
        
    Returns:
        a list of actions
    """
    if selection <= 0:
        return [ACTION.STOP]

    candidate = candidates.loc[selection - 1]
    target_view_index = candidate['view_index']
    navigable_index = candidate['navigable_index']

    heading_id = view_index % 12
    elevation_id = view_index // 12
    target_heading_id = target_view_index % 12
    target_elevation_id = target_view_index // 12

    actions = []
    while True:
        if elevation_id < target_elevation_id:
            actions.append(ACTION.UP)
            elevation_id += 1
        elif elevation_id > target_elevation_id:
            actions.append(ACTION.DOWN)
            elevation_id -= 1
        elif heading_id != target_heading_id:
            target_heading_id = (target_heading_id + 12 - heading_id) % 12
            heading_id = 0
            if target_heading_id <= 6:
                actions += [ACTION.RIGHT] * target_heading_id
            else:
                actions += [ACTION.LEFT] * (12 - target_heading_id)
            heading_id = target_heading_id
        else:
            actions.append(ACTION.MOVE(navigable_index))
            break
    return actions
