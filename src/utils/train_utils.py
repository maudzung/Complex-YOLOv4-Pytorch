"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: utils functions that use for training process
"""

import copy
import os

import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist


def create_optimizer(configs, model):
    """Create optimizer for training process"""
    if hasattr(model, 'module'):
        train_params = [param for param in model.module.parameters() if param.requires_grad]
    else:
        train_params = [param for param in model.parameters() if param.requires_grad]

    if configs.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(train_params, lr=configs.lr / configs.batch_size / 4., momentum=configs.momentum,
                                    weight_decay=configs.weight_decay)
    elif configs.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(train_params, lr=configs.lr / configs.batch_size / 4.,
                                     weight_decay=configs.weight_decay, betas=(0.9, 0.999), eps=1e-08)
    else:
        assert False, "Unknown optimizer type"

    return optimizer


def create_lr_scheduler(optimizer, configs):
    """Create learning rate scheduler for training process"""

    def burnin_schedule(i):
        if i < configs.burn_in:
            factor = pow(i / configs.burn_in, 4)
        elif i < configs.steps[0]:
            factor = 1.0
        elif i < configs.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    lr_scheduler = LambdaLR(optimizer, burnin_schedule)

    return lr_scheduler


def get_saved_state(model, optimizer, lr_scheduler, epoch, configs):
    """Get the information to save with checkpoints"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    saved_state = {
        'epoch': epoch,
        'configs': configs,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': lr_scheduler.state_dict(),
        'state_dict': model_state_dict,
    }

    return saved_state


def save_checkpoint(checkpoints_dir, saved_fn, saved_state, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
    save_path = os.path.join(checkpoints_dir, '{}_epoch_{}.pth'.format(saved_fn, epoch))

    torch.save(saved_state, save_path)
    print('save a checkpoint at {}'.format(save_path))


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def get_tensorboard_log(model):
    tensorboard_log = {}
    if hasattr(model, 'module'):
        yolo_layers = model.module.yolo_layers
    else:
        yolo_layers = model.yolo_layers
    for j, yolo_layer in enumerate(yolo_layers):
        for name, metric in yolo_layer.metrics.items():
            if j == 0:
                tensorboard_log['{}'.format(name)] = metric
            else:
                tensorboard_log['{}'.format(name)] += metric

    return tensorboard_log
