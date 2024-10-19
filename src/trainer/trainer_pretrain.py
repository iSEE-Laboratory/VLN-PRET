import math
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from itertools import cycle
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

import utils
from .trainer_base import TrainerBase


class TrainerPretrain(TrainerBase):
    def __init__(self, datasets, agent, tasks):
        assert 'MLM' in tasks  # at least one pretraining task.

        super().__init__(datasets['MLM']['train'], None, agent)
        self.tasks = tasks
        self.datasets = datasets

    def _get_dataloader(self, split, batch_size):
        dataloaders = dict()
        if 'MLM' in self.tasks:
            dataset = self.datasets['MLM'][split]
            sampler = DistributedSampler(dataset) if self.distributed else RandomSampler(dataset)
            dataloaders['MLM'] = DataLoader(dataset, batch_size, sampler=sampler, num_workers=4, collate_fn=collate_fn_MLM)

        if 'ITM' in self.tasks:
            dataset = self.datasets['ITM'][split]
            sampler = DistributedSampler(dataset) if self.distributed else RandomSampler(dataset)
            dataloaders['ITM'] = DataLoader(dataset, batch_size, sampler=sampler, num_workers=4, collate_fn=collate_fn_ITM)

        dataloader = EndlessDataLoader(dataloaders)
        return dataloader

    def _train_n_iteration(self, dataloader, n):
        self.agent.train()

        loss_dict = pd.Series(dtype=float)
        task_count = pd.Series(dtype=float)

        print_progress = utils.PrintProgress('TRAIN: ', total=n)
        for i in range(n):
            dataset_name, batch = next(dataloader)
            loss, temp_loss_log = self.agent(dataset_name, batch)

            for task, loss_ in temp_loss_log.items():
                loss_dict[task] = loss_dict.get(task, 0) + loss_
                task_count[task] = task_count.get(task, 0) + 1

            self._optimize(loss)
            if self.rank == 0:
                print_progress(postfix=(loss_dict/task_count).round(2).to_dict())
        self.lr_scheduler.step()
        return (loss_dict / task_count).round(4).to_dict()

    def _reduce(self, result):
        if isinstance(result, dict):
            object_list = [None] * self.world_size
            dist.all_gather_object(object_list, result)
            if self.rank == 0:
                for key in result:
                    items = [result_[key] for result_ in object_list]
                    result[key] = sum(items) / self.world_size
            return result
        else:
            raise NotImplementedError()

    def train(self, batch_size, iteration_num, log_every, evaluate_first):
        if self.distributed:
            dist.barrier()

        epoch_num = iteration_num / math.ceil(self.train_set_size // (batch_size * self.world_size))
        print(f'Start training {iteration_num} iteration({epoch_num:.2f} epoch).')

        train_loader = self._get_dataloader('train', batch_size)
        train_loader = iter(train_loader)

        best_accuracy = -1
        for iteration in range(0, iteration_num, log_every):
            self.log('', no_prefix=True)  # empty line
            self.log(f'{iteration}-{iteration + log_every} / {iteration_num}')

            # train
            loss_dict = self._train_n_iteration(train_loader, n=log_every)
            loss_dict = self._reduce(loss_dict) if self.distributed else loss_dict
            self.log(f'losses\n{pd.Series(loss_dict)}')
            if self.log_dir is not None:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'loss/{key}', value, iteration)

            # evaluate
            for split in ['val_seen', 'val_unseen']:
                result = self.evaluate(split, batch_size * 4)
                result = self._reduce(result) if self.distributed else result
                self.log(f'{split}: {str(result)}')
                if self.log_dir is not None:
                    for task, accuracy in result.items():
                        self.writer.add_scalar(f'{split}/accuracy_{task}', accuracy, iteration)

                if split == 'val_unseen' and result['MLM'] > best_accuracy:
                    self.log('*** best up to now ***')
                    best_accuracy = result['MLM']
                    if self.log_dir is not None:
                        agent = self.agent.module if self.distributed else self.agent
                        agent.save(self.log_dir)

    @torch.no_grad()
    def evaluate(self, split, batch_size):
        agent = self.agent.module if self.distributed else self.agent
        agent.eval()
        result = dict()

        val_loader = self._get_dataloader(split, batch_size)
        for dataset_name, loader in val_loader.loaders.items():
            if dataset_name == 'MLM':
                total_num_MLM, correct_num_MLM = 0, 0
                for batch in loader:
                    prediction_MLM, target_MLM = agent(dataset_name, batch)
                    total_num_MLM += len(prediction_MLM) if prediction_MLM is not None else 0
                    correct_num_MLM += (prediction_MLM == target_MLM).sum().item()
                result['MLM'] = correct_num_MLM / total_num_MLM

            elif dataset_name == 'ITM':
                total_num_ITM, correct_num_ITM = 0, 0
                for batch in loader:
                    prediction_ITM, target_ITM = agent(dataset_name, batch)
                    total_num_ITM += len(prediction_ITM)
                    correct_num_ITM += (prediction_ITM == target_ITM).sum().item()
                result['ITM'] = correct_num_ITM / total_num_ITM

            elif dataset_name == 'SAP_SAR':
                total_num, correct_num = 0, 0
                result['SAR'] = 0
                for batch in loader:
                    prediction, target, loss_SAR = agent(dataset_name, batch)
                    total_num += len(prediction)
                    correct_num += (prediction == target).sum().item()
                    result['SAR'] += loss_SAR * len(prediction)
                result['SAP'] = correct_num / total_num
                result['SAR'] /= total_num
                if 'SAP' not in self.tasks:
                    del result['SAP']
                if 'SAR' not in self.tasks:
                    del result['SAR']
        return result


def collate_fn_MLM(batch_list):
    path_id_list, text_id_list, \
        history_panorama_list, history_length_list, \
        history_candidates_list, history_teacher_selection_list = zip(*batch_list)
    path_id = torch.tensor(path_id_list)
    text_ids = pad_sequence(text_id_list, batch_first=True)  # (B, text_length)
    history_panorama = torch.cat(history_panorama_list, dim=0)  # (B * history_length, 36, 768)
    history_length = torch.tensor(history_length_list)  # (B)
    return path_id, text_ids, history_panorama, history_candidates_list, history_teacher_selection_list, history_length


collate_fn_ITM = collate_fn_MLM


def collate_fn_SAP_SAR(batch_list):
    text_id_list, graph_list, path_list, next_nodes = zip(*batch_list)
    
    text_ids = pad_sequence(text_id_list, batch_first=True)  # (B, text_length)
    next_nodes = list(next_nodes)
    graph_list = list(graph_list)
    path_list = list(path_list)
    return text_ids, graph_list, path_list, next_nodes


class EndlessDataLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.iters = {key: iter(value) for key, value in loaders.items()}
        self.epochs = {key: 0 for key, value in loaders.items()}
        self.tasks = cycle(loaders.keys())

    def __iter__(self):
        while True:
            task = next(self.tasks)
            batch_iter = self.iters[task]
            try:
                batch = next(batch_iter)
            except StopIteration:
                self.epochs[task] += 1
                if isinstance(self.loaders[task].sampler, DistributedSampler):
                    self.loaders[task].sampler.set_epoch(self.epochs[task])

                batch_iter = iter(self.loaders[task])
                batch = next(batch_iter)
                self.iters[task] = batch_iter
            yield task, batch
