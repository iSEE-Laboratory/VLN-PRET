import numpy as np
import torch.optim as optim


class LRScheduler:
    def __init__(self, optimizer, max_step, warmup_ratio=0, fn='none'):
        assert fn in ('none', 'linear', 'cosine')
        assert warmup_ratio < 1

        self.optimizer = optimizer

        self.cur_step = 0
        self.max_step = max_step
        self.warmup_step = int(warmup_ratio * max_step) + 1
        # 加一是为了实现简单, 可以多调用一次self.step修改初始学习率

        self.slope = 1 / (max_step - self.warmup_step + 1)
        self.fn = self._get_fn(fn)

        self.initial_lr = [param_group['lr'] for param_group in self.optimizer.param_groups]
        if self.warmup_step > 0:
            self.step()

    def step(self):
        self.cur_step = min(self.cur_step + 1, self.max_step)
        if self.cur_step < self.warmup_step:
            scale = self.cur_step / self.warmup_step
        else:
            x = self.slope * (self.cur_step - self.warmup_step)
            scale = self.fn(x)

        for i in range(len(self.optimizer.param_groups)):
            lr = self.initial_lr[i] * scale
            self.optimizer.param_groups[i]['lr'] = lr
    
    def get_lr(self):
        lr_list = [param_group['lr'] for param_group in self.optimizer.param_groups]
        return lr_list

    def _get_fn(self, fn):
        if fn == 'none':
            return lambda x: 1
        elif fn == 'linear':
            return lambda x: 1 - x
        elif fn == 'cosine':
            return lambda x: (np.cos(np.pi * x) + 1) / 2
