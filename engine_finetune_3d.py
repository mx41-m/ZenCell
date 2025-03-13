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
from typing import Iterable, Optional, Callable

import torch
import torch.nn.functional as F
import torch.distributed as dist

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train(model: torch.nn.Module,
          data_loader: Iterable, optimizer: torch.optim.Optimizer,
          max_norm: float, log_writer=None, args=None) -> None:
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter("epoch", misc.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    print_freq = 20

    accum_iter = args.accum_iter
    steps_per_epoch = len(data_loader.dataset) / data_loader.batch_size / dist.get_world_size()

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (
        input_vol, label_bin, label_flow
    ) in enumerate(metric_logger.log_every(data_loader, print_freq)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / steps_per_epoch, args)

        input_vol = input_vol.cuda(non_blocking=True)
        label_bin = label_bin.cuda(non_blocking=True)
        label_flow = label_flow.cuda(non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
                outputs = model(input_vol)
        outputs = outputs.float()
        mask_loss = F.binary_cross_entropy_with_logits(
            outputs[:, 0, :, :, :], label_bin.float(), reduction="mean",
        )
        flow_loss = F.mse_loss(outputs[:, 1:, :, :, :], label_flow * 5., reduction="mean")

        loss = mask_loss + flow_loss
        loss_value = loss.item()
        mask_loss_value = mask_loss.item()
        flow_loss_value = flow_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss.backward()
        optimizer.step()
        
        if (data_iter_step + 1) % accum_iter == 0:
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mask_loss=mask_loss_value)
        metric_logger.update(flow_loss=flow_loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(epoch=data_iter_step / steps_per_epoch)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        mask_loss_value_reduce = misc.all_reduce_mean(mask_loss_value)
        flow_loss_value_reduce = misc.all_reduce_mean(flow_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(data_iter_step / steps_per_epoch * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('mask_loss', mask_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('flow_loss', flow_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
