from datetime import datetime


class PrintProgress:
    def __init__(self, desc, total):
        self.step = 1
        self.desc = desc
        self.total = total
        self.max_len = 0

    def format_time(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

    def __call__(self, step=None, postfix=''):
        if step is None:
            step = self.step
            self.step += 1
        if type(postfix) == dict:
            postfix = ', '.join([f'{key}={postfix[key]}' for key in postfix])

        now = datetime.now()
        if step == 1:
            self.start_time = now
            speed = '? it/s'
            time_cost = '00:00:00'
            time_remain = '?'
        else:
            speed = 1 / (now - self.last_time).total_seconds()
            time_cost = self.format_time((now - self.start_time).total_seconds())
            time_remain = self.format_time((self.total - step) / speed)
            if speed < 1:
                speed = f'{1 / speed:.2f}s/it'
            else:
                speed = f'{speed:.2f}it/s'
        time_cost = time_cost.removeprefix('00:')
        time_remain = time_remain.removeprefix('00:')
        progress = int(round(step * 100 / self.total))
        status = f'\r{self.desc}{progress}% {step}/{self.total} [{time_cost}<{time_remain}, {speed}, {postfix}]'
        
        self.last_time = now
        if self.max_len > len(status):
            status += ' ' * (self.max_len - len(status))
        else:
            self.max_len = len(status)
        if step == self.total:
            print(status)
        else:
            print(status, end='')
