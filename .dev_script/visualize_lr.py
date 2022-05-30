import argparse
import json
import os.path as osp
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine import Config, Runner
from mmengine.registry import DATASETS, MODELS


@MODELS.register_module()
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)

    def forward(self, data_batch, return_loss=False):
        return dict(log_vars={})


@DATASETS.register_module()
class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore

    def __init__(self, data_len):
        self.data = torch.randn(data_len, 2)
        self.label = torch.ones(data_len)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize the learning rate and momentum of given config'
        ', and this script will overwrite the log_config')
    parser.add_argument(
        '--config', default='./config.py', help='train config file path')
    parser.add_argument(
        '--work-dir',
        default='./work_dir',
        help='the dir to save logs and models')
    parser.add_argument(
        '--window-size',
        default='12*14',
        help='Size of the window to display images, in format of "$W*$H".')
    args = parser.parse_args()
    return args


def optimize_step(self, runner, batch_idx, data_batch, outputs):
    pass


def runtimeinfo_step(self, runner, batch_idx, data_batch=None):

    def get_momentum():
        group = runner.optimizer.param_groups[0]
        if 'momentum' in group.keys():
            momentum = group['momentum']
        elif 'betas' in group.keys():
            momentum = group['betas'][0]
        else:
            momentum = 0
        return momentum

    runner.message_hub.update_info('iter', runner.iter)
    runner.message_hub.update_scalar('train/lr',
                                     runner.optimizer.param_groups[0]['lr'])
    runner.message_hub.update_scalar('train/momentum', get_momentum())


@patch('torch.cuda.is_available', lambda: False)
@patch('mmengine.hooks.OptimizerHook.after_train_iter', optimize_step)
@patch('mmengine.hooks.RuntimeInfoHook.before_train_iter', runtimeinfo_step)
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg['window_size'] = args.window_size
    cfg['work_dir'] = args.work_dir
    extra_cfg = dict(
        model=dict(type='ToyModel'),
        train_dataloader=dict(
            dataset=dict(type='ToyDataset', data_len=cfg.num_iters),
            sampler=dict(type='DefaultSampler'),
            batch_size=1,
            num_workers=0),
        optimizer=dict(type='SGD', lr=0.1, momentum=0.9),
        env_cfg=dict(dist_cfg=dict(backend='nccl')),
        default_hooks=dict(
            logger=dict(type='LoggerHook', interval=cfg.interval),
            checkpoint=None),
        log_level='INFO')

    cfg.merge_from_dict(extra_cfg)
    runner = Runner.from_cfg(cfg)
    runner.train()
    plot_lr_curve(runner, cfg)


def plot_lr_curve(runner, cfg):
    json_file = osp.join(cfg.work_dir, runner.timestamp,
                         'vis_data/scalars.json')
    data_dict = dict(LearningRate=[], Momentum=[])
    assert osp.isfile(json_file)
    with open(json_file) as f:
        for line in f:
            log = json.loads(line.strip())
            data_dict['LearningRate'].append(log['lr'])
            data_dict['Momentum'].append(log['momentum'])

    wind_w, wind_h = (int(size) for size in cfg.window_size.split('*'))
    _, axes = plt.subplots(2, 1, figsize=(wind_w, wind_h))
    font_size = 20

    plt.subplots_adjust(hspace=0.5)
    for index, (updater_type, data_list) in enumerate(data_dict.items()):
        ax = axes[index]
        if cfg.train_cfg.by_epoch is True:
            ax.plot(data_list)
            ax.xaxis.tick_top()
            ax.set_xlabel('Iters', fontsize=font_size)
            ax.xaxis.set_label_position('top')
            sec_ax = ax.secondary_xaxis(
                'bottom',
                functions=(lambda x: x / cfg.num_iters * cfg.interval,
                           lambda y: y * cfg.num_iters / cfg.interval))
            sec_ax.tick_params(labelsize=font_size)
            sec_ax.set_xlabel('Epochs', fontsize=font_size)
        else:
            x_list = np.arange(len(data_list)) * cfg.interval
            ax.plot(x_list, data_list)
            ax.set_xlabel('Iters', fontsize=font_size)
        ax.set_ylabel(updater_type, fontsize=font_size)
        if updater_type == 'LearningRate':
            title = 'No learning rate scheduler'
            for schuduler in cfg.param_scheduler:
                if schuduler.type.endswith('LR'):
                    title = schuduler.type
        else:
            title = 'No momentum scheduler'
            for schuduler in cfg.param_scheduler:
                if schuduler.type.endswith('Momentum'):
                    title = schuduler.type
        ax.set_title(title, fontsize=font_size)
        ax.grid()
        ax.tick_params(labelsize=font_size)
    save_path = osp.join(cfg.work_dir, runner.timestamp,
                         'visualization-result')
    plt.savefig(save_path)
    print(f'The learning rate graph is saved at {save_path}.png')


if __name__ == '__main__':
    main()
