import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import torch
from torchvision.models import resnet18
from torch.optim.lr_scheduler import LambdaLR
from easydict import EasyDict as edict


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


if __name__ == '__main__':
    configs = edict()
    configs.burn_in = 500
    configs.steps = [300, 400]
    net = resnet18()
    optimizer = torch.optim.SGD(net.parameters(), 0.0025)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
    scheduler = create_lr_scheduler(optimizer, configs)
    for i in range(500):
        print(i, scheduler.get_lr())
        scheduler.step()
