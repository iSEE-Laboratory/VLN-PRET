import os
import signal
import os.path as osp

from datetime import datetime


class Logger:
    def __init__(self, log_dir=None):
        self.format = '%Y-%m-%d %H:%M:%S: '
        self.file = None  # 用于写入文件
        self.log_dir = log_dir

        if log_dir is not None:
            path = osp.join(log_dir, 'log.txt')
            if not osp.exists(log_dir):
                os.mkdir(log_dir)
            self.file = open(path, 'a')
    
    def __get_prefix(self):
        return datetime.now().strftime(self.format)

    def log(self, message, no_prefix=False):
        """
        在终端和文件内输出信息(如果指定了文件目录)
        """
        if not no_prefix:
            message = self.__get_prefix() + message
        print(message)
        if self.file is not None:
            print(message, file=self.file, flush=True)

    def set_interupt_message(self, message):
        """
        设置Ctrl-C中断时输出的内容
        """
        def handler(signum, frame):
            nonlocal message
            self.log(message)
            exit()
        signal.signal(signal.SIGINT, handler)
