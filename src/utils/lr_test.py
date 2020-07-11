import math
from copy import copy
import os

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import torch
from torchvision.models import resnet18
from torch.optim.lr_scheduler import LambdaLR
from easydict import EasyDict as edict
import matplotlib.pyplot as plt


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
        lf = lambda x: (((1 + math.cos(x * math.pi / configs.epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
        plot_lr_scheduler(optimizer, lr_scheduler, configs.epochs, save_dir=configs.log_dir)
    else:
        raise ValueError

    return lr_scheduler


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'LR.png'), dpi=200)


if __name__ == '__main__':
    configs = edict()
    configs.burn_in = 500
    configs.steps = [300, 400]
    configs.lr_type = 'cosin'  # multi_step
    configs.log_dir = '../../logs/'
    configs.epochs = 300
    net = resnet18()
    optimizer = torch.optim.SGD(net.parameters(), 0.01)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
    scheduler = create_lr_scheduler(optimizer, configs)
    for i in range(configs.epochs):
        print(i, scheduler.get_lr())
        scheduler.step()
