import os
import yaml
import torch
import argparse
import os.path as osp
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime

import utils
import agent as agent_factory
import model as model_factory
from dataset import *
from trainer import *


def get_dataset_agent(args):
    # datasets
    if args.dataset == 'R2R':
        datasets = {split: R2R(split=split) for split in ['train', 'aug', 'val_train', 'val_seen', 'val_unseen', 'test']}
    elif args.dataset == 'RxR':
        datasets = {split: RxR(split=split) for split in ['train', 'aug', 'val_seen', 'val_unseen', 'test']}
    elif args.dataset == 'R2RPretrain':
        datasets = {
            'MLM': {split: R2R_MLMDataset(split=split, use_aug_data=(split == 'train')) for split in ['train', 'val_seen', 'val_unseen']},
            'ITM': {split: R2R_ITMDataset(split=split, use_aug_data=(split == 'train')) for split in ['train', 'val_seen', 'val_unseen']},
        }
    elif args.dataset == 'RxRPretrain':
        datasets = {
            'MLM': {split: RxR_MLMDataset(split=split, use_aug_data=(split == 'train')) for split in ['train', 'val_seen', 'val_unseen']},
            'ITM': {split: RxR_ITMDataset(split=split, use_aug_data=(split == 'train')) for split in ['train', 'val_seen', 'val_unseen']},
        }

    # models
    model = getattr(model_factory, args.model)(args)

    # agents, used to adapt model to different datasets or training methods
    agent = getattr(agent_factory, args.agent)(args, model)

    return datasets, agent


@utils.print_exception
def main(rank, world_size, args):
    utils.setup_seed(args.seed)
    print(f'RANK {rank}: Setup seed {args.seed}.')

    # set device for distributed training
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # create dataset, model and agent
    print('Loading dataset.')
    datasets, agent = get_dataset_agent(args)

    # load trained agent
    if args.load:
        print('Loading model from', args.load)
        agent.load(args.load, strict=not args.not_load_strict)

    # create optimizer
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(agent.parameters(), lr=args.lr, weight_decay=0.0)

    # create scheduler
    scheduler = utils.LRScheduler(
        optimizer,
        max_step=args.iteration_num // args.log_every,
        warmup_ratio=args.warmup_ratio,
        fn=args.lr_scheduler)

    # create Trainer, use different trainer to support different task
    if args.trainer == 'TF_SF':
        assert args.dataset in {'R2R', 'RxR'}
        main_metric = {'R2R': 'SPL', 'RxR': 'sDTW'}[args.dataset]
        trainer = TrainerTFSF(datasets, agent, optimizer, scheduler, main_metric)
    elif args.trainer == 'Pretrain':
        assert args.dataset in {'R2RPretrain', 'RxRPretrain'}
        trainer = TrainerPretrain(datasets, agent, optimizer, scheduler, args.tasks)

    if args.use_amp:
        trainer.set_use_amp()

    if world_size > 1:
        trainer.set_distributed(rank, world_size)

    if args.log and rank == 0:
        time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_dir = osp.join('log', args.method, f'{time}_{args.description}')
        trainer.set_log_dir(log_dir)  # create log dir
        with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f, allow_unicode=True)
            from config import FEATURE_PATH
            f.write('feature path: %s' % FEATURE_PATH)

    # start training
    trainer.train(args.batch_size, args.iteration_num, args.log_every, args.evaluate_first)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='distributed training: --gpu 0,1,2')
    parser.add_argument('--log', action='store_true', help='save training data or not')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_every', type=int, default=1000)

    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--description', type=str, required=True)
    parser.add_argument('--load', type=str, default='', help='path to the dir of checkpoint model')
    parser.add_argument('--not_load_strict', default=False, action='store_true')

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--trainer', type=str, required=True)

    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--iteration_num', type=int, required=True)
    parser.add_argument('--evaluate_first', default=False, action='store_true')

    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, default='none')
    parser.add_argument('--use_amp', default=False, action='store_true')

    parser.add_argument('--text_backbone', type=str, default='ALBEF', choices=['ALBEF', 'CLIP', 'BERT'])
    parser.add_argument('--tasks', type=str, default='', nargs='+', help='pretraining tasks')
    parser.add_argument('--dropout', type=float, default=0, help='feature dropout')
    parser.add_argument('--OPE_layer_num', type=int, default=2)
    parser.add_argument('--MAM_layer_num', type=int, default=4)
    parser.add_argument('--CCM_layer_num', type=int, default=1)
    parser.add_argument('--use_panorama', default=False, action='store_true')
    parser.add_argument('--use_directed', default=False, action='store_true')

    parser.add_argument('--max_step', type=int, default=15)
    parser.add_argument('--loss_weight', type=float, default=0.5)
    parser.add_argument('--mask_visited', default=False, action='store_true')

    args = parser.parse_args()
    assert args.iteration_num % args.log_every == 0

    gpus = args.gpu.split(',')
    world_size = len(gpus)

    if not torch.cuda.is_available():
        print('Training with CPU')
        main(0, world_size, args)
    elif world_size == 1:
        print(f'Training with a single GPU cuda:{args.gpu}')
        device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
        torch.cuda.set_device(device)
        main(0, world_size, args)
    else:
        print('Training with GPUs', args.gpu)
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(utils.get_available_port())
            mp.spawn(main, (world_size, args), nprocs=world_size)
        except KeyboardInterrupt:  # kill all subprocess by ctrl+c
            for p in mp.active_children():
                p.kill()
        print('KeyboardInterrupt')
