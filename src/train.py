import time
import numpy as np
import sys
import random
import os
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append('./')

from data_process.kitti_dataloader import create_train_val_dataloader
from models.model_utils import create_model, load_pretrained_model, make_data_parallel, resume_model, get_num_parameters
from train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from train_utils import reduce_tensor, to_python_float
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.config import parse_configs
from utils.evaluation_utils import non_max_suppression_rotated_bbox, get_batch_statistics_rotated_bbox, ap_per_class


def main():
    configs = parse_configs()

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None

    # model
    model = create_model(configs)

    # Data Parallel
    model = make_data_parallel(model, configs)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)

    # optionally load weight from a checkpoint
    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx, configs.overwrite_global_2_local)
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))

    # optionally resume from a checkpoint
    if configs.resume_path is not None:
        checkpoint = resume_model(configs.resume_path, configs.arch, configs.gpu_idx)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        configs.start_epoch = checkpoint['epoch'] + 1

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_loader, val_loader, train_sampler = create_train_val_dataloader(configs)
    if logger is not None:
        logger.info('number of batches in train set: {}'.format(len(train_loader)))
        if val_loader is not None:
            logger.info('number of batches in val set: {}'.format(len(val_loader)))

    if configs.evaluate:
        assert val_loader is not None, "The validation should not be None"
        eval_metrics = evaluate_one_epoch(val_loader, model, configs.start_epoch - 1, configs, logger)
        precision, recall, AP, f1, ap_class = eval_metrics
        print('Evaluate - precision: {}, recall: {}, AP: {}, f1: {}, ap_class: {}'.format(precision, recall, AP, f1,
                                                                                          ap_class))
        return

    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        # Get the current learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}] learning rate: {:.2e}'.format(epoch, configs.num_epochs, lr))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train_one_epoch(train_loader, model, optimizer, epoch, configs, logger, tb_writer)
        if not configs.no_val:
            precision, recall, AP, f1, ap_class = evaluate_one_epoch(val_loader, model, epoch, configs, logger)
            val_metrics_dict = {'precision': precision, 'recall': recall, 'AP': AP, 'f1': f1, 'ap_class': ap_class}
            if tb_writer is not None:
                tb_writer.add_scalars('Validation', val_metrics_dict, epoch)

        # Save checkpoint
        if configs.is_master_node and ((epoch % configs.checkpoint_freq) == 0):
            saved_state = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, epoch)

        # Adjust learning rate
        lr_scheduler.step()

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_loader, model, optimizer, epoch, configs, logger, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    num_iters_per_epoch = len(train_loader)

    # switch to train mode
    model.train()
    start_time = time.time()
    for batch_idx, batch_data in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - start_time)
        _, imgs, targets = batch_data
        global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1

        batch_size = imgs.size(0)

        targets = targets.to(configs.device, non_blocking=True)
        imgs = imgs.to(configs.device, non_blocking=True)
        total_loss, outputs = model(imgs, targets)

        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)

        # zero the parameter gradients
        optimizer.zero_grad()
        # compute gradient and perform backpropagation
        total_loss.backward()
        optimizer.step()

        if configs.distributed:
            reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
        else:
            reduced_loss = total_loss.data
        losses.update(to_python_float(reduced_loss), batch_size)
        # measure elapsed time
        torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)

        # Tensorboard
        tensorboard_log = {}
        for j, yolo_layer in enumerate(model.yolo_layers):
            for name, metric in yolo_layer.metrics.items():
                if j == 0:
                    tensorboard_log['{}'.format(name)] = metric
                else:
                    tensorboard_log['{}'.format(name)] += metric

        tensorboard_log['avg_loss'] = losses.avg

        if tb_writer is not None:
            tb_writer.add_scalars('Train', tensorboard_log, global_step)

        # Log message
        if logger is not None:
            if ((batch_idx + 1) % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))

        start_time = time.time()


def evaluate_one_epoch(val_loader, model, epoch, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    conf_thres = 0.5
    nms_thres = 0.5
    iou_threshold = 0.5

    progress = ProgressMeter(len(val_loader), [batch_time, data_time, losses],
                             prefix="Evaluate - Epoch: [{}/{}]".format(epoch, configs.num_epochs))
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            batch_size = imgs.size(0)

            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] *= configs.img_size

            outputs = model(imgs)
            outputs = non_max_suppression_rotated_bbox(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=iou_threshold)

            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.gpu_idx is None):
                total_loss = torch.mean(total_loss)

            if configs.distributed:
                reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
            else:
                reduced_loss = total_loss.data
            losses.update(to_python_float(reduced_loss), batch_size)
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
