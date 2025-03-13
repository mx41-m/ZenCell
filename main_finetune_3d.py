# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.lr_decay as lrd
import util.misc as misc

import models_vit

from engine_finetune_3d import train
from dataset.finetune_dataset_3d import FinetuneDataset3D
from cellpose.dynamics import compute_masks


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--data_split", type=str, default="train", choices=["train", "trainval"])

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=50, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    model = models_vit.__dict__[args.model](
        drop_path_rate=args.drop_path,
        out_chans=4,
    )

    if args.data_split == "train":
        dataset_train = FinetuneDataset3D(
            data_root=args.data_root,
            split="train",
            roi_size=model.patch_embed_roi.img_size,
            ctx_size=model.patch_embed_ctx.img_size,
            alignment=model.patch_embed_ctx.patch_size,
        )
    elif args.data_split == "trainval":
        dataset_train = torch.utils.data.ConcatDataset([
            FinetuneDataset3D(
                data_root=args.data_root,
                split=split,
                roi_size=model.patch_embed_roi.img_size,
                ctx_size=model.patch_embed_ctx.img_size,
                alignment=model.patch_embed_ctx.patch_size,
            )
            for split in ["train", "val"]
        ])
    else:
        raise NotImplementedError()
    assert len(dataset_train) >= 1
    sampler_train = []
    target_sampler_len = eff_batch_size * (
        (len(dataset_train) * args.epochs - 1) // eff_batch_size + 1
    )
    for i in range((target_sampler_len - 1) // len(dataset_train) + 1):
        g = torch.Generator()
        g.manual_seed(i)
        sampler_train.extend(torch.randperm(len(dataset_train), generator=g).tolist())
    sampler_train = sampler_train[misc.get_rank(): target_sampler_len: misc.get_world_size()]
    print(f"Target epochs {args.epochs}, actual epochs {target_sampler_len / len(dataset_train):.2f}")

    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        # interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    no_weight_decay_list = []
    for name, param in model_without_ddp.named_parameters():
        if param.ndim == 1 or name.endswith(".bias"):
            no_weight_decay_list.append(name)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=no_weight_decay_list,
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train(model, data_loader_train, optimizer, args.clip_grad, log_writer, args)
    if args.output_dir:
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=args.epochs - 1,
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
