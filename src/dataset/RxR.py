import torch
import jsonlines
import pandas as pd
import os.path as osp
import torch.distributed as dist
from torch.utils.data import BatchSampler, RandomSampler, DistributedSampler

from .R2R import R2REnvBatch
from config import RXR_DIR

RxREnvBatch = R2REnvBatch


class RxR:
    def __init__(self, split, language=['en', 'hi', 'te'], render=False, img_size=(640, 480), vfov=60):
        self.vfov = vfov
        self.split = split
        self.render = render
        self.img_size = img_size

        if 'en' in language:
            language.append('en-IN')
            language.append('en-US')
        if 'hi' in language:
            language.append('hi-IN')
        if 'te' in language:
            language.append('te-IN')
        self.language = set(language)
        self.dataset = load_RxR_dataset(split, language)  # pd.DataFrame

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
            batch.reset_index(inplce=True)
        env = RxREnvBatch(batch, render=self.render, img_size=self.img_size, vfov=self.vfov)
        return batch, env

    def get_batches(self, batch_size):
        if dist.is_initialized():
            sampler = DistributedSampler(self.dataset, shuffle=True, drop_last=False)
            sampler.set_epoch(self.epoch)
        else:
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            sampler = RandomSampler(self.dataset, generator=generator)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)

        for batch_ids in batch_sampler:
            batch = self.dataset.loc[batch_ids].reset_index()
            env = RxREnvBatch(batch, render=self.render, img_size=self.img_size, vfov=self.vfov)
            yield batch, env
        self.epoch += 1


def load_RxR_dataset(split, language):
    if split == 'test':
        path = osp.join(RXR_DIR, f'rxr_test_standard_public_guide.jsonl')
    elif split == 'aug':
        path = osp.join(RXR_DIR, f'rxr_marky_train_guide.jsonl')
    else:
        path = osp.join(RXR_DIR, f'rxr_{split}_guide.jsonl')
    print(f'Loading from {path}')

    dataset = []
    with jsonlines.open(path) as reader:
        for line in reader:
            if line['language'] in language:
                sample = dict()
                sample['scan'] = line['scan']
                sample['path'] = line['path']
                if split != 'test':
                    sample['path_id'] = line['path_id']
                sample['heading'] = line['heading']
                sample['language'] = line['language']
                sample['instruction'] = line['instruction']
                sample['instruction_id'] = line['instruction_id']
                dataset.append(sample)
    return pd.DataFrame(dataset)
