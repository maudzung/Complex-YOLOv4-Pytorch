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

from data_process.kitti_dataloader import create_train_val_dataloader, create_test_dataloader
from models.model_utils import create_model, load_pretrained_model, make_data_parallel, resume_model, get_num_parameters
from train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from train_utils import reduce_tensor, to_python_float
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.config import parse_configs


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
    best_val_loss = np.inf
    earlystop_count = 0
    is_best = False

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
        best_val_loss = checkpoint['best_val_loss']
        earlystop_count = checkpoint['earlystop_count']
        configs.start_epoch = checkpoint['epoch'] + 1

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_loader, val_loader, train_sampler = create_train_val_dataloader(configs)
    test_loader = create_test_dataloader(configs)
    if logger is not None:
        logger.info('number of batches in train set: {}'.format(len(train_loader)))
        if val_loader is not None:
            logger.info('number of batches in val set: {}'.format(len(val_loader)))
        logger.info('number of batches in test set: {}'.format(len(test_loader)))

    if configs.evaluate:
        assert val_loader is not None, "The validation should not be None"
        val_loss = evaluate_one_epoch(val_loader, model, configs.start_epoch - 1, configs, logger)
        print('Evaluate, val_loss: {}'.format(val_loss))
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
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, configs, logger, tb_writer)
        loss_dict = {'train': train_loss}
        if not configs.no_val:
            val_loss = evaluate_one_epoch(val_loader, model, epoch, configs, logger, tb_writer)
            is_best = val_loss <= best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            loss_dict['val'] = val_loss

        if not configs.no_test:
            test_loss = evaluate_one_epoch(test_loader, model, epoch, configs, logger, tb_writer)
            loss_dict['test'] = test_loss
        # Write tensorboard
        if tb_writer is not None:
            tb_writer.add_scalars('Loss', loss_dict, epoch)
        # Save checkpoint
        if configs.is_master_node and (is_best or ((epoch % configs.checkpoint_freq) == 0)):
            saved_state = get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss,
                                          earlystop_count)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best, epoch)
        # Check early stop training
        if configs.earlystop_patience is not None:
            earlystop_count = 0 if is_best else (earlystop_count + 1)
            print_string = ' |||\t earlystop_count: {}'.format(earlystop_count)
            if configs.earlystop_patience <= earlystop_count:
                print_string += '\n\t--- Early stopping!!!'
                break
            else:
                print_string += '\n\t--- Continue training..., earlystop_count: {}'.format(earlystop_count)
            if logger is not None:
                logger.info(print_string)
        # Adjust learning rate
        if configs.lr_type == 'plateau':
            assert (not configs.no_val), "Only use plateau when having validation set"
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_loader, model, optimizer, epoch, configs, logger, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    # switch to train mode
    model.train()
    start_time = time.time()
    for batch_idx, batch_data in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - start_time)
        _, imgs, targets = batch_data

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
        for j, yolo in enumerate(model.yolo_layers):
            for name, metric in yolo.metrics.items():
                if j == 0:
                    tensorboard_log['train/{}'.format(name)] = metric
                else:
                    tensorboard_log['train/{}'.format(name)] += metric

        # Log message
        if logger is not None:
            if ((batch_idx + 1) % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))

        start_time = time.time()

    return losses.avg


def evaluate_one_epoch(val_loader, model, epoch, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time, losses],
                             prefix="Evaluate - Epoch: [{}/{}]".format(epoch, configs.num_epochs))
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            batch_size = resized_imgs.size(0)
            target_seg = target_seg.to(configs.device, non_blocking=True)
            resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
            pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model.inference(
                resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg)

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

    return losses.avg


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
