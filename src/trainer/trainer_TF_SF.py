import math
import json
import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch.distributed as dist
from datetime import datetime
from torch.cuda.amp import autocast

import utils
from dataset import connectivity
from .trainer_base import TrainerBase


class TrainerTFSF(TrainerBase):
    def __init__(self, datasets, agent, optimizer, lr_scheduler, main_metric='SPL'):
        assert main_metric in ('SPL', 'sDTW')

        super().__init__(datasets['train'], None, agent, optimizer, lr_scheduler)
        self.main_metric = main_metric
        self.datasets = datasets

    def _get_dataloader(self, split, batch_size):
        if split not in self.datasets:
            return None, 0

        dataset = self.datasets[split]
        if split in {'train', 'aug'}:
            dataloader = iter(EndlessGenerator(dataset, batch_size))
        elif split in {'val_train', 'val_seen', 'val_unseen', 'test'}:
            dataloader = dataset.get_batches(batch_size)

        dataloader_length = np.ceil(np.ceil(len(dataset) / self.world_size) / batch_size)
        return dataloader, dataloader_length

    def _train_n_iteration(self, train_loader, aug_loader, n):
        self.agent.train()

        loss_dict = 0
        print_progress = utils.PrintProgress(desc='TRAIN: ', total=n)

        loaders = [train_loader] if aug_loader is None else [train_loader, aug_loader]
        for i in range(n // len(loaders)):
            for loader in loaders:
                try:
                    batch, env = next(loader)
                    loss, loss_log_temp = self.agent(batch, env)
                    self._optimize(loss)
                    loss_dict = loss_dict + loss_log_temp
                    if self.rank == 0:
                        print_progress(postfix=(loss_dict/(i+1)).round(2).to_dict())
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        print('======SKIP======')
                        continue
                    else:
                        raise
        self.lr_scheduler.step()
        return (loss_dict/(i+1)).to_dict()  # average loss of this epoch

    def _reduce(self, result):
        if isinstance(result, dict):  # reduce loss_dict
            object_list = [None] * self.world_size
            dist.all_gather_object(object_list, result)
            if self.rank == 0:
                for key in result:
                    items = [result_[key] for result_ in object_list]
                    result[key] = sum(items) / self.world_size
            return result
        elif isinstance(result, list):  # gather evaluation result
            object_list = [None] * self.world_size
            dist.all_gather_object(object_list, result)
            if self.rank == 0:
                result = sum(object_list, start=[])  # sum all list to a single list
            return result
        else:
            raise NotImplementedError()

    def _evaluate_first(self, evaluator, batch_size):
        val_start_time = datetime.now()
        for split in ['val_seen', 'val_unseen']:
            if split not in self.datasets:
                continue

            val_loader, length = self._get_dataloader(split, batch_size)
            all_result = self.evaluate(val_loader, length)  # list[dict]
            all_result = self._reduce(all_result) if self.distributed else all_result
            metrics = evaluator.evaluate(all_result)  # dict

            message = ', '.join(f'{key} {value: .4f}' for key, value in metrics.items())
            self.log(f'{split}: {message}')
            if self.rank == 0 and self.log_dir is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar(f'{split}/{key}', value, 0)
        self.log(f'EVALUATING time: {str(datetime.now() - val_start_time)}')

    def train(self, batch_size, iteration_num, log_every, evaluate_first):
        if self.distributed:
            dist.barrier()
        
        train_loader, _ = self._get_dataloader('train', batch_size)
        aug_loader, _ = self._get_dataloader('aug', batch_size)

        evaluator = Evaluator({split: dataset for split, dataset in self.datasets.items() if 'val' in split})

        epoch_num = iteration_num / (1 + int(aug_loader is not None)) / math.ceil(self.train_set_size // (batch_size * self.world_size))
        print(f'Start training {iteration_num} iteration({epoch_num:.2f} epoch).')

        start_time = datetime.now()
        best_iteration, best_result = -1, None
        for iteration in range(0, iteration_num + log_every, log_every):
            if iteration == 0:
                if evaluate_first:
                    self._evaluate_first(evaluator, batch_size)
                continue

            self.log('', no_prefix=True)  # empty line
            self.log(f'{iteration - log_every}-{iteration} / {iteration_num}')

            train_start_time = datetime.now()
            with autocast(enabled=self.use_amp):
                loss_dict = self._train_n_iteration(train_loader, aug_loader, n=log_every)
            loss_dict = self._reduce(loss_dict) if self.distributed else loss_dict
            self.log(f'TRAIN time: {str(datetime.now() - train_start_time)}')
            self.log(f'losses\n{pd.Series(loss_dict)}')
            if self.log_dir is not None and self.rank == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'loss/{key}', value, iteration)

            # evaluate
            split_result = dict()
            val_start_time = datetime.now()
            for split in ['val_train', 'val_seen', 'val_unseen']:
                if split not in self.datasets:
                    continue
                
                val_loader, length = self._get_dataloader(split, batch_size)
                all_result = self.evaluate(val_loader, length)  # list[dict]
                all_result = self._reduce(all_result) if self.distributed else all_result
                split_result[split] = all_result
                metrics = evaluator.evaluate(all_result)  # dict

                message = ', '.join(f'{key} {value: .4f}' for key, value in metrics.items())
                self.log(f'{split}: {message}')
                if self.rank == 0:
                    if self.log_dir is not None:
                        for key, value in metrics.items():
                            self.writer.add_scalar(f'{split}/{key}', value, iteration)

                    better = (best_result is None or metrics[self.main_metric] > best_result[self.main_metric])
                    if split == 'val_unseen' and better:
                        best_iteration = iteration
                        best_result = pd.Series(metrics)  # convert to pd.Series for pretty print
                        self.log('*** best up to now ***')
                        self.logger.set_interupt_message(f'\n---finished training---\nbest iteration: {best_iteration}\nbest result:\n{best_result}')
            self.log(f'EVALUATING time: {str(datetime.now() - val_start_time)}')

            # save model and evaluation result
            if iteration == best_iteration and self.rank == 0 and self.log_dir is not None:
                # save evaluation result, so that it can be submited to the website
                for split, all_result in split_result.items():
                    save_path = osp.join(self.log_dir, f'trajectory_{split}.json')
                    with open(save_path, 'w') as f:
                        json.dump(all_result, f)

                agent = self.agent.module if self.distributed else self.agent
                agent.save(self.log_dir)
            # break
        self.log(f'---finished training---\n'
            f'best iteration: {best_iteration}\n'
            f'best result:\n{best_result}')
        self.log(f'Time consuming: {str(datetime.now() - start_time)}')

        # testing
        if self.log_dir is not None and self.rank == 0:
            # load the best model
            agent = self.agent.module if self.distributed else self.agent
            agent.load(self.log_dir)

            print('\n\nStart testing')
            test_loader, length = self._get_dataloader('test', batch_size)
            all_result = self.evaluate(test_loader, length)
            all_result = self._reduce(all_result) if self.distributed else all_result
            save_path = osp.join(self.log_dir, f'trajectory_test.json')
            with open(save_path, 'w') as f:
                json.dump(all_result, f)
            print('Finished testing')

    @torch.no_grad()
    def evaluate(self, val_loader, length):
        """
        It's strange that the performance sligthly changes with different batch size even with the same model.
        When evaluation with different batch size, the result changes a slightly, especially for the longer path.
        After debugging, it seems to be a precision problem.
        At each step, the value after the decimal point has changed in the sixth or seventh digit when the batch size changes.
        According to https://pytorch.org/docs/stable/notes/numerical_accuracy.html, this is due to the pytorch implementation.
        The batched computation is not guaranteed the same.

        Submit URL: https://eval.ai/web/challenges/challenge-page/97/submission
        Submit Format:
            a json file like:
            [
                {
                    "instr_id": path_id_string + "_" + instruction_index_string,
                    "trajectory":[
                        [viewpoint_id_string, heading_radians_float, elevation_radians_float],
                        ...
                    ]
                },
                ...
            ]
        """
        agent = self.agent.module if self.distributed else self.agent
        agent.eval()

        all_result = []

        if self.rank == 0:
            print_progress = utils.PrintProgress(desc='EVALUATE: ', total=length)
        for batch, env in val_loader:
            batch_size = len(batch)
            batch_result = []
            for i in range(batch_size):
                instr_id = batch.at[i, 'instruction_id']
                if isinstance(instr_id, np.int64):  # 原本是numpy.int64的类型
                    instr_id = int(instr_id)
                result = {
                    'instr_id': instr_id,
                    'trajectory': []
                }
                batch_result.append(result)

            # initialize
            step = 0
            agent.reset(batch)
            stopped = np.zeros(batch_size, dtype=np.bool_)

            # start navigation
            obs = env.reset()
            while agent.step < agent.max_step:  # NOTE: agent.step is different from step if the agent use graph decision
                for i in range(batch_size):
                    if step > 0 and actions[i] == -1:  # temporary stopped
                        continue
                    if stopped[i]:
                        continue
                    batch_result[i]['trajectory'].append(
                        (obs.at[i, 'viewpoint_id'], obs.at[i, 'heading'], obs.at[i, 'elevation'])
                    )

                actions = agent.take_action(obs)
                stopped = stopped | (actions == 0)
                actions[stopped] = 0  # avoid agents move after STOP

                obs = env.step_viewpoint(obs, actions)
                step += 1
                if stopped.all():
                    break
            all_result.extend(batch_result)

            if self.rank == 0:
                print_progress()
        return all_result




class EndlessGenerator:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __iter__(self):
        while True:
            iterator = self.dataset.get_batches(self.batch_size)
            yield from iterator




class Evaluator:
    def __init__(self, datasets):
        self.error_margin = 3.0
        self.metrics = ['NE', 'PL', 'SR', 'SPL', 'ONE', 'OSR', 'nDTW', 'sDTW']

        self.ndtw = utils.ndtw_initialize()
        self.instruction_id_to_sample = dict()
        for dataset in datasets.values():
            for i in range(len(dataset)):
                sample, env = dataset[i]
                instruction_id = sample['instruction_id']
                if isinstance(instruction_id, np.int64):  # RxR, 原本是numpy.int64的类型
                    instruction_id = int(instruction_id)
                scan_id = sample['scan']
                path = sample['path']
                self.instruction_id_to_sample[instruction_id] = (scan_id, path)

    def __check_adjacent(self, scan_id, node1, node2):
        if node1 == node2:
            return
        try:
            graph = connectivity.get_graph(scan_id)
            graph[node1][node2]  # get edge
        except KeyError:
            msg = (f'The provided trajectory moves from {node1} to {node2} but the navigation graph contains no '
                'edge between these viewpoints. Please ensure the provided navigation trajectories '
                'are valid, so that trajectory length can be accurately calculated.')
            raise RuntimeError(msg)

    def _evaluate_single(self, item):
        instruction_id = item['instr_id']
        trajectory = [viewpoint_id for viewpoint_id, _, _ in item['trajectory']]
        final_position = trajectory[-1]

        scan_id, gt_path = self.instruction_id_to_sample[instruction_id]
        start = gt_path[0]
        goal = gt_path[-1]
        assert trajectory[0] == start

        metrics = dict()

        metrics['NE'] = connectivity.get_distance(scan_id, final_position, goal)
        metrics['ONE'] = min(connectivity.get_distance(scan_id, node, goal) for node in trajectory)

        path_length = 0
        prev = trajectory[0]
        for i in range(len(trajectory)):
            curr = trajectory[i]
            self.__check_adjacent(scan_id, prev, curr)
            path_length += connectivity.get_distance(scan_id, prev, curr)
            prev = curr
        metrics['PL'] = path_length

        metrics['SR'] = int(metrics['NE'] < self.error_margin)
        metrics['OSR'] = int(metrics['ONE'] < self.error_margin)

        shortest_path_length = connectivity.get_distance(scan_id, start, goal)
        metrics['SPL'] = metrics['SR'] * shortest_path_length / max(shortest_path_length, path_length)

        metrics['nDTW'] = self.ndtw[scan_id](trajectory, gt_path, metric='ndtw')
        metrics['sDTW'] = metrics['SR'] * metrics['nDTW']

        return metrics

    def evaluate(self, results):
        """
        Args:
            results: list[dict]
        Returns:
            DataFrame, each row is a sample, each columns is a metric
        """
        num = len(results)

        metrics = dict()
        for item in results:
            temp = self._evaluate_single(item)
            for metric in self.metrics:
                metrics[metric] = metrics.get(metric, 0) + temp[metric]
        for metric in metrics:
            metrics[metric] /= num
        return metrics
