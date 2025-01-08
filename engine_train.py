# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
import time

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
#from timm.data import Mixup
#from timm.utils import accuracy

import util.misc as misc

import util.lr_sched as lr_sched
from util.metric import calculate_rmse_psnr
from util.noise_project import *

from util.optimizer_surv import get_optimizer
from util.warmup_scheduler_surv import WarmupScheduler

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad() # Gradient clear for all parameters

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    # #这里会enumerate 1305个，相当于 是在一个epoch中会有1305个patch进行迭代    2023704
    for data_iter_step, (imglr, imghr, value_min, value_max) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler

        #TODO
        imglr = imglr.to(device, non_blocking=True)
        imglrNoUpscale = imglr
        #1020 2023 self
        if args.inputzoom:
            imglr = F.interpolate(imglr, scale_factor=args.scale, mode='bicubic',align_corners=True)
        imghr = imghr.to(device, non_blocking=True)
        value_min = value_min.to(device).detach()
        value_max = value_max.to(device).detach()

        if args.useNoisyProjection:
            with torch.no_grad():
                totNoisePerElem = (2 * (args.input_size ** 2)) ** (1 / 2)
                nsPwr = totNoisePerElem / args.scale ** 2
                nsEps = noisePowerNormalize(nsPwr,value_min,value_max,args.up).detach().reshape(-1, 1)

        # with torch.cuda.amp.autocast():
            outputs = model(imglr)
            fakehr = projectToNoiseLevel(outputs, imglrNoUpscale, nsEps, args.scale, args.scale)
        else:
            fakehr = model(imglr)
            
        loss = criterion(fakehr, imghr)
        rmse, psnr, nrmse, nrmse_Tnorm = calculate_rmse_psnr(fakehr,imghr)
        # loss.backward()
        # optimizer.step()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if accum_iter>1:
            loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if data_iter_step % accum_iter == 128 and data_iter_step % 256 == 0:

            #lr_sched.adjust_learning_rate(optimizer, epoch, args)
            lr_scheduler.step(epoch)
            
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(data_iter_step / len(data_loader) + epoch)
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/psnr', psnr, epoch_1000x)
            log_writer.add_scalar('train/nrmse_Tnorm', nrmse_Tnorm, epoch_1000x)
            log_writer.add_scalar('train/lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model,criterion, device, args, epoch = None, stage = "test",log_writer=None): # 2023704
    criterion = criterion

    metric_logger = misc.MetricLogger(delimiter=" ")
    if stage == "test":
        header = 'Test'
    else:
        header = "Val"
    accum_iter = args.accum_iter
    # switch to evaluation mode
    model.eval()
    nrmse_all_down = 0.
    nrmse_all_up = 0.
    nrmse_all_down_denorm = 0.
    nrmse_all_up_denorm = 0.


    for data_iter_step, (imglr, imghr, value_min, value_max) in enumerate(metric_logger.log_every(data_loader, 100, header = header)):

        imglr = imglr.to(device, non_blocking=True)
        imglrNoUpscale = imglr
        #1020 2023 self
        if args.inputzoom:
            imglr = F.interpolate(imglr, scale_factor=args.scale, mode='bicubic',align_corners=True)
        imghr = imghr.to(device, non_blocking=True)
        value_min = value_min.to(device).detach()
        value_max = value_max.to(device).detach()

        # compute output
        # with torch.cuda.amp.autocast()
        if args.useNoisyProjection:
            totNoisePerElem = (2 * (args.input_size ** 2)) ** (1 / 2)
            nsPwr = totNoisePerElem / args.scale ** 2
            nsEps = noisePowerNormalize(nsPwr, value_min, value_max, args.up).reshape(-1, 1)

            outputs = model(imglr)
            fakehr = projectToNoiseLevel(outputs, imglrNoUpscale, nsEps, args.scale, args.scale)
            loss = criterion(fakehr, imghr)

            rmse, psnr, nrmse, nrmse_Tnorm = calculate_rmse_psnr(fakehr,imghr)

            nrmse_all_up += torch.norm(fakehr - imghr) ** 2
            nrmse_all_down += torch.norm(imghr) ** 2
            nrmse_all_up_denorm += torch.norm(noisePowerDeNormalize(fakehr - imghr, value_min, value_max, args.up)) ** 2
            nrmse_all_down_denorm += torch.norm(noisePowerDeNormalize(imghr, value_min, value_max, args.up)) ** 2
        else:
            fakehr = model(imglr)
            loss = criterion(fakehr, imghr)
            rmse, psnr, nrmse, nrmse_Tnorm= calculate_rmse_psnr(fakehr,imghr)

            nrmse_all_up += torch.norm(fakehr - imghr) ** 2
            nrmse_all_down += torch.norm(imghr) ** 2
            nrmse_all_up_denorm += torch.norm(denormalize(fakehr, value_min, value_max, args.up, args.down) -
                                              denormalize(imghr, value_min, value_max, args.up, args.down)
                                              ) ** 2
            nrmse_all_down_denorm += torch.norm(denormalize(imghr, value_min, value_max, args.up, args.down)) ** 2

        batch_size = imglr.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['rmse'].update(rmse.item(), n=batch_size)
        metric_logger.meters['psnr'].update(psnr.item(), n=batch_size)
        metric_logger.meters['nrmse'].update(nrmse.item(), n=batch_size)
        metric_logger.meters['nrmse_Tnorm'].update(nrmse_Tnorm.item(), n=batch_size)


    # gather the stats from all processes
    nrmse_all = torch.sqrt(nrmse_all_up) / torch.sqrt(nrmse_all_down)
    nrmse_all_denorm = torch.sqrt(nrmse_all_up_denorm) / torch.sqrt(nrmse_all_down_denorm)
    metric_logger.update(nrmse_all=nrmse_all.item())
    metric_logger.update(nrmse_all_denorm=nrmse_all_denorm.item())
    metric_logger.update(nrmse_Tnorm=nrmse_Tnorm.item())
    if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(data_iter_step / len(data_loader) + epoch)
            log_writer.add_scalar(f'{header}/loss', loss, epoch_1000x)
            log_writer.add_scalar(f'{header}/psnr', psnr, epoch_1000x)
            log_writer.add_scalar(f'{header}/nrmse_Tnorm', nrmse_all_denorm, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print('* RMSE {rmse.global_avg:.5f} PSNR {psnr.global_avg:.5f} NRMSE {nrmse.global_avg:.5f} loss {losses.global_avg:.5f}'
          'nrmse_all {nrmse_all.global_avg:.5f} nrmse_all_denorm {nrmse_all_denorm.global_avg:.5f}'
          .format(rmse=metric_logger.rmse, psnr=metric_logger.psnr, nrmse = metric_logger.nrmse, losses=metric_logger.loss,
                  nrmse_all=metric_logger.nrmse_all, nrmse_all_denorm=metric_logger.nrmse_all_denorm))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}