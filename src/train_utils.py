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
import math

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR
import torch.distributed as dist


def create_optimizer(configs, model):
    """Create optimizer for training process"""
    if hasattr(model, 'module'):
        train_params = [param for param in model.module.parameters() if param.requires_grad]
    else:
        train_params = [param for param in model.parameters() if param.requires_grad]

    if configs.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(train_params, lr=configs.lr, momentum=configs.momentum,
                                    weight_decay=configs.weight_decay)
    elif configs.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(train_params, lr=configs.lr, weight_decay=configs.weight_decay)
    else:
        assert False, "Unknown optimizer type"

    return optimizer


def create_lr_scheduler(optimizer, configs):
    """Create learning rate scheduler for training process"""
    if configs.lr_type == 'step_lr':
        lr_scheduler = StepLR(optimizer, step_size=configs.lr_step_size, gamma=configs.lr_factor)
    elif configs.lr_type == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=configs.lr_factor, patience=configs.lr_patience)
    elif configs.optimizer_type == 'cosin':
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / configs.num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
        lr_scheduler.last_epoch = configs.start_epoch - 1  # do not move
        # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        # plot_lr_scheduler(optimizer, scheduler, epochs)
    else:
        raise TypeError

    return lr_scheduler


def get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss, earlystop_count):
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
        'best_val_loss': best_val_loss,
        'earlystop_count': earlystop_count,
    }

    return saved_state


def save_checkpoint(checkpoints_dir, saved_fn, saved_state, is_best, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
    if is_best:
        save_path = os.path.join(checkpoints_dir, '{}_best.pth'.format(saved_fn))
    else:
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
