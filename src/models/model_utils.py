"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.18
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: utils functions that use for model
"""

import sys

import torch

sys.path.append('../')

from models.darknet2pytorch import Darknet


def create_model(configs):
    """Create model based on architecture name"""
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = Darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)
    else:
        assert False, 'Undefined model backbone'

    return model


def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters


def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.batch_size = int(configs.batch_size / configs.ngpus_per_node)
            configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs.gpu_idx is not None:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model


if __name__ == '__main__':
    import argparse

    from torchsummary import summary
    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='../config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')

    configs = edict(vars(parser.parse_args()))

    configs.device = torch.device('cuda:1')

    model = create_model(configs).to(device=configs.device)
    sample_input = torch.randn((1, 3, 608, 608)).to(device=configs.device)
    # summary(model.cuda(), (3, 608, 608))
    output = model(sample_input, targets=None)
    print(output.size())
