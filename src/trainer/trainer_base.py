import math
import torch
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader

import utils


class TrainerBase:
    def __init__(self, train_set, validate_set, agent, optimizer, lr_scheduler):
        self.train_set = train_set
        self.validate_set = validate_set
        self.train_set_size = len(train_set)

        self.agent = agent
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.log_dir = None
        self.writer = None
        self.logger = utils.Logger(None)
    
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.agent.to(self.device)

        self.use_amp = False

    def set_use_amp(self):
        self.use_amp = True
        self.scaler = GradScaler()

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir)
            self.logger = utils.Logger(log_dir)
    
    def set_distributed(self, rank, world_size):
        self.distributed = True
        self.rank = rank
        self.world_size = world_size

        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", init_method=None, world_size=world_size, rank=rank)
        self.agent = DDP(self.agent, device_ids=[rank], find_unused_parameters=True)
        print(f'RANK {self.rank}: Setup distributed training.')

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5, error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5, error_if_nonfinite=True)
            self.optimizer.step()

    def log(self, message, no_prefix=False):
        if self.rank != 0:
            return

        self.logger.log(message, no_prefix)

    def _get_dataloader(self, dataset, batch_size):
        sampler = DistributedSampler(dataset) if self.distributed else RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        return dataloader

    def _train_n_iteration(self, train_loader, n):
        self.agent.train()

        loss_dict = 0
        print_progress = utils.PrintProgress(desc='TRAIN: ', total=n)
        for i in range(n):
            x, y = next(train_loader)
            x = x.to(self.device)
            y = y.to(self.device)
            loss, loss_log_temp = self.agent(x, y, loss=True)  # temp_loss_log is a pd.Series
            self._optimize(loss)
            loss_dict = loss_dict + loss_log_temp

            if self.rank == 0:
                print_progress(postfix=(loss_dict/(i+1)).round(2).to_dict())
        self.lr_scheduler.step()
        return (loss_dict/(i+1)).to_dict()  # average loss of this epoch

    def _reduce(self, result):
        if isinstance(result, dict):
            object_list = [None] * self.world_size
            dist.all_gather_object(object_list, result)
            if self.rank == 0:
                for key in result:
                    items = [result_[key] for result_ in object_list]
                    result[key] = sum(items) / self.world_size
            return result
        elif isinstance(result, torch.Tensor):
            dist.all_reduce(result)
            result = result / self.world_size
            return result
        else:
            raise NotImplementedError()

    def train(self, batch_size, iteration_num, log_every):
        if self.distributed:
            dist.barrier()
        
        epoch_num = iteration_num / math.ceil(self.train_set_size // (batch_size * self.world_size))
        print(f'Start training {iteration_num} iteration({epoch_num:.2f} epoch).')

        train_loader = self._get_dataloader(self.train_set, batch_size)
        train_loader = iter(utils.EndlessDataLoader(train_loader))
        validate_loader = self._get_dataloader(self.validate_set, batch_size)

        best, best_iteration, best_result = -1, -1, -1
        for iteration in range(0, iteration_num, log_every):
            self.log('', no_prefix=True)  # empty line
            self.log(f'{iteration}-{iteration + log_every} / {iteration_num}')

            # train
            loss_dict = self._train_n_iteration(train_loader, n=log_every)
            loss_dict = self._reduce(loss_dict) if self.distributed else loss_dict
            self.log(f'losses\n{pd.Series(loss_dict)}')

            # evaluate
            result = self.evaluate(validate_loader)  # DataFrame
            result = self._reduce(result) if self.distributed else result

            # log
            curr = result.item()
            if self.rank == 0:
                self.log(f'accuracy: {curr: .4f}')
                if curr > best:
                    best = curr
                    best_iteration = iteration
                    best_result = result
                    self.log('*** best up to now ***')
                    self.logger.set_interupt_message(  # print this message when input CTRL+C
                        f'---finished training---\n'
                        f'best iteration: {best_iteration}\n'
                        f'best result:\n{best_result:.4f}')
                if self.log_dir is not None:
                    self.writer.add_scalars('train/loss', loss_dict, iteration)
                    self.writer.add_scalar('evaluate/accuracy', curr, iteration)
                    if iteration == best_iteration:
                        if self.distributed:
                            self.agent.module.save(self.log_dir)
                        else:
                            self.agent.save(self.log_dir)

            # synchronize
            if self.distributed:
                dist.barrier()
        self.log(f'---finished training---\n'
            f'best iteration: {best_iteration}\n'
            f'best result:\n{best_result:.4f}')

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.agent.eval()
        if self.rank == 0:
            dataloader = tqdm(dataloader, 'EVALUATING')

        correct_num = 0
        total_num = 0
        for (img_batch, label_batch) in dataloader:
            batch_size = len(img_batch)
            img_batch = img_batch.to(self.device)
            label_batch = label_batch.to(self.device)

            logits = self.agent(img_batch)
            predictions = logits.argmax(dim=1)

            correct_num += (predictions == label_batch).sum()
            total_num += batch_size
        return correct_num / total_num
