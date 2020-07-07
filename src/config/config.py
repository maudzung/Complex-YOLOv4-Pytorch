"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: The configurations of the project will be defined here
"""

import os
import argparse
import sys

import torch
import numpy as np
from easydict import EasyDict as edict

sys.path.append('../')

from utils.misc import make_folder


def parse_configs():
    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')
    parser.add_argument('--seed', type=int, default=2020,
                        help='re-produce the results with seed random')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolo', metavar='FN',
                        help='The name using for saving logs, models,...')
    ####################################################################
    ##############     Model configs            ###################
    ####################################################################
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default=None, metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--dropout_p', type=float, default=0.5, metavar='P',
                        help='The dropout probability of the model')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--multiscale_training', action='store_true',
                        help='If true, use scaling data for training')
    parser.add_argument('--no-val', action='store_true',
                        help='If true, dont evaluate the model on the val set')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size (default: 64), this is the total'
                             'batch size of all GPUs on the current node when using'
                             'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--subdivisions', type=int, default=16,
                        help='subdivisions during training')
    parser.add_argument('--print_freq', type=int, default=20, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--tensorboard_freq', type=int, default=20, metavar='N',
                        help='frequency of saving tensorboard (default: 10)')
    parser.add_argument('--checkpoint_freq', type=int, default=5, metavar='N',
                        help='frequency of saving checkpoints (default: 3)')
    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################

    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=300, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=0.00261, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--minimum_lr', type=float, default=1e-7, metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('--momentum', type=float, default=0.949, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4, metavar='WD',
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--optimizer_type', type=str, default='adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam')
    parser.add_argument('--lr_type', type=str, default='plateau', metavar='SCHEDULER',
                        help='the type of the learning rate scheduler (steplr or ReduceonPlateau)')
    parser.add_argument('--burn_in', type=int, default=1000, metavar='N',
                        help='number of burn in step')
    parser.add_argument('--steps', nargs='*', default=[400000, 450000],
                        help='number of burn in step')

    ####################################################################
    ##############     Loss weight            ###################
    ####################################################################

    ####################################################################
    ##############     Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--world-size', default=-1, type=int, metavar='N',
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='N',
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    parser.add_argument('--evaluate', action='store_true',
                        help='only evaluate the model, not training')
    parser.add_argument('--resume_path', type=str, default=None, metavar='PATH',
                        help='the path of the resumed checkpoint')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the image of testing phase will be saved')

    ####################################################################
    ##############     Demonstration configurations     ###################
    ####################################################################
    parser.add_argument('--video_path', type=str, default=None, metavar='PATH',
                        help='the path of the video that needs to demo')
    parser.add_argument('--output_format', type=str, default='text', metavar='PATH',
                        help='the type of the demo output')
    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_demo_output', action='store_true',
                        help='If true, the image of demonstration phase will be saved')

    configs = edict(vars(parser.parse_args()))

    ####################################################################
    ############## Hardware configurations ############################
    ####################################################################
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    configs.ngpus_per_node = torch.cuda.device_count()

    configs.pin_memory = True

    ####################################################################
    ##############     Data configs            ###################
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    ####################################################################
    ############## logs, Checkpoints, and results dir ########################
    ####################################################################
    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.saved_fn)
    configs.logs_dir = os.path.join(configs.working_dir, 'logs', configs.saved_fn)

    configs.results_dir = os.path.join(configs.working_dir, 'results')

    make_folder(configs.checkpoints_dir)
    make_folder(configs.logs_dir)
    make_folder(configs.results_dir)

    if configs.save_test_output:
        configs.saved_dir = os.path.join(configs.results_dir, configs.saved_fn)
        make_folder(configs.saved_dir)

    if configs.save_demo_output:
        configs.save_demo_dir = os.path.join(configs.results_dir, 'demo', configs.saved_fn)
        make_folder(configs.save_demo_dir)

    configs.img_size = 608

    return configs


if __name__ == "__main__":
    configs = parse_configs()
    print(configs)
