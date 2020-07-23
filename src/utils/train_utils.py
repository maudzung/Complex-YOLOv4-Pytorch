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
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
import matplotlib.pyplot as plt


def create_optimizer(configs, model):
    """Create optimizer for training process
    Refer from https://github.com/ultralytics/yolov3/blob/e80cc2b80e3fd46395e8ec75f843960100927ff2/train.py#L94
    """
    if hasattr(model, 'module'):
        params_dict = dict(model.module.named_parameters())
    else:
        params_dict = dict(model.named_parameters())

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in params_dict.items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif ('conv' in k) and ('.weight' in k):
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if configs.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(pg0, lr=configs.lr, momentum=configs.momentum, nesterov=True)
    elif configs.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(pg0, lr=configs.lr)
    else:
        assert False, "Unknown optimizer type"

    optimizer.add_param_group({'params': pg1, 'weight_decay': configs.weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))

    return optimizer


def create_lr_scheduler(optimizer, configs):
    """Create learning rate scheduler for training process"""

    if configs.lr_type == 'multi_step':
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
    elif configs.lr_type == 'cosin':
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / configs.num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
        # plot_lr_scheduler(optimizer, lr_scheduler, configs.num_epochs, save_dir=configs.logs_dir)
    else:
        raise ValueError

    return lr_scheduler


def get_saved_state(model, optimizer, lr_scheduler, epoch, configs):
    """Get the information to save with checkpoints"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    utils_state_dict = {
        'epoch': epoch,
        'configs': configs,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': copy.deepcopy(lr_scheduler.state_dict())
    }

    return model_state_dict, utils_state_dict


def save_checkpoint(checkpoints_dir, saved_fn, model_state_dict, utils_state_dict, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
    model_save_path = os.path.join(checkpoints_dir, 'Model_{}_epoch_{}.pth'.format(saved_fn, epoch))
    utils_save_path = os.path.join(checkpoints_dir, 'Utils_{}_epoch_{}.pth'.format(saved_fn, epoch))

    torch.save(model_state_dict, model_save_path)
    torch.save(utils_state_dict, utils_save_path)

    print('save a checkpoint at {}'.format(model_save_path))


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
    if hasattr(model, 'module'):
        yolo_layers = model.module.yolo_layers
    else:
        yolo_layers = model.yolo_layers

    tensorboard_log = {}
    tensorboard_log['Average_All_Layers'] = {}
    for idx, yolo_layer in enumerate(yolo_layers, start=1):
        layer_name = 'YOLO_Layer{}'.format(idx)
        tensorboard_log[layer_name] = {}
        for name, metric in yolo_layer.metrics.items():
            tensorboard_log[layer_name]['{}'.format(name)] = metric
            if idx == 1:
                tensorboard_log['Average_All_Layers']['{}'.format(name)] = metric / len(yolo_layers)
            else:
                tensorboard_log['Average_All_Layers']['{}'.format(name)] += metric / len(yolo_layers)

    return tensorboard_log


def plot_lr_scheduler(optimizer, scheduler, num_epochs=300, save_dir=''):
    # Plot LR simulating training for full num_epochs
    optimizer, scheduler = copy.copy(optimizer), copy.copy(scheduler)  # do not modify originals
    y = []
    for _ in range(num_epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, num_epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'LR.png'), dpi=200)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from torchvision.models import resnet18

    configs = edict()
    configs.burn_in = 500
    configs.steps = [300, 400]
    configs.lr_type = 'cosin'  # multi_step
    configs.logs_dir = '../../logs/'
    configs.num_epochs = 300
    net = resnet18()
    optimizer = torch.optim.Adam(net.parameters(), 0.001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
    scheduler = create_lr_scheduler(optimizer, configs)
    for i in range(configs.num_epochs):
        # print(i, scheduler.get_lr())
        scheduler.step()
