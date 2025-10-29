#!/usr/bin/env python

import numpy as np
import zarr
import ast
import os
import torch
from tqdm import tqdm
from cellpose.dynamics import compute_masks
import argparse

import models_vit


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--chns", type=lambda x: tuple(int(y) for y in x.split(",")), nargs="+", required=True)
parser.add_argument("--zarr_path", type=str, required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NPROCS"])
torch.cuda.set_device(rank % torch.cuda.device_count())

model = models_vit.__dict__[args.model](out_chans=4)
model.half().eval().cuda()
model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])

z, y, x = 640, 1024, 1024
zarr_fp = zarr.open(args.zarr_path, mode="r")
zmax, ymax, xmax = zarr_fp["0"].shape[2:]
task_list = []
for ref_chn, sig_chn in args.chns:
    for z0 in range(0, zmax, z):
        for y0 in range(0, ymax, y):
            for x0 in range(0, xmax, x):
                task_list.append((ref_chn, sig_chn, z0, y0, x0))
if rank == 0:
    print(f"Total task: {len(task_list)}")

rz, ry, rx = model.patch_embed_roi.img_size
cz, cy, cx = model.patch_embed_ctx.img_size
az, ay, ax = model.patch_embed_ctx.patch_size
assert z % rz == 0 and y % ry == 0 and x % rx == 0
zpad_head, ypad_head, xpad_head = (cz - rz) // 2 // az * az, (cy - ry) // 2 // ay * ay, (cx - rx) // 2 // ax * ax
zpad_tail, ypad_tail, xpad_tail = cz - rz - zpad_head, cy - ry - ypad_head, cx - rx - xpad_head


def crop_with_pad(chn, z0, y0, x0):
    z_st, z_ed = z0 - zpad_head - rz // 2, z0 + z + zpad_tail + rz // 2
    y_st, y_ed = y0 - ypad_head - ry // 2, y0 + y + ypad_tail + ry // 2
    x_st, x_ed = x0 - xpad_head - rx // 2, x0 + x + xpad_tail + rx // 2
    z_slice = slice(max(z_st, 0), min(z_ed, zmax))
    y_slice = slice(max(y_st, 0), min(y_ed, ymax))
    x_slice = slice(max(x_st, 0), min(x_ed, xmax))
    z_pad = (max(-z_st, 0), max(z_ed - zmax, 0))
    y_pad = (max(-y_st, 0), max(y_ed - ymax, 0))
    x_pad = (max(-x_st, 0), max(x_ed - xmax, 0))
    arr = zarr_fp["0"][chn, z_slice, y_slice, x_slice]
    #arr = np.cast["<u2"](zarr_fp["0"][0, chn, z_slice, y_slice, x_slice]) ### for different data format
    if any(a > 0 or b > 0 for a, b in (z_pad, y_pad, x_pad)):
        arr = np.pad(arr, (z_pad, y_pad, x_pad))
    return arr


for ref_chn, sig_chn, z0, y0, x0 in (tqdm if rank == 0 else lambda x: x)(task_list[rank::world_size]):
    ref_arr = crop_with_pad(ref_chn, z0, y0, x0)
    sig_arr = crop_with_pad(sig_chn, z0, y0, x0)
    
    z_offs_list, y_offs_list, x_offs_list = np.meshgrid(np.arange(0, z + 1, rz // 2), np.arange(0, y + 1, ry // 2), np.arange(0, x + 1, rx // 2))
    z_offs_list, y_offs_list, x_offs_list = z_offs_list.reshape(-1).tolist(), y_offs_list.reshape(-1).tolist(), x_offs_list.reshape(-1).tolist()
    
    cell_prob = torch.zeros([z + rz, y + ry, x + rx], device="cuda")
    cell_flow = torch.zeros([3, z + rz, y + ry, x + rx], device="cuda")
    weight = torch.zeros([z + rz, y + ry, x + rx], device="cuda")
    z_dist_sq = (torch.linspace(start=0., end=2., steps=rz) - 1.) ** 2
    y_dist_sq = (torch.linspace(start=0., end=2., steps=ry) - 1.) ** 2
    x_dist_sq = (torch.linspace(start=0., end=2., steps=rx) - 1.) ** 2
    weight_template = (3. - z_dist_sq.view(-1, 1, 1) - y_dist_sq.view(1, -1, 1) - x_dist_sq.view(1, 1, -1)) / 3.
    weight_template = weight_template.cuda()
    
    with torch.no_grad():
        ref_arr = torch.from_numpy(ref_arr).cuda()
        sig_arr = torch.from_numpy(sig_arr).cuda()
        for z_offs, y_offs, x_offs in zip(z_offs_list, y_offs_list, x_offs_list):
            ref_slice = ref_arr[z_offs: z_offs + cz, y_offs: y_offs + cy, x_offs: x_offs + cx]
            sig_slice = sig_arr[z_offs: z_offs + cz, y_offs: y_offs + cy, x_offs: x_offs + cx]
            input_vol = torch.stack([ref_slice, sig_slice], dim=0).float()
            assert input_vol.size() == (2, cz, cy, cx)
            input_vol = input_vol - input_vol.mean(dim=(1, 2, 3), keepdim=True)
            input_vol = input_vol / (input_vol.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
            input_vol = input_vol.half()
            slice_pred = model(input_vol[None])[0]
            assert slice_pred.size()[1:] == (rz, ry, rx)
            cell_prob[z_offs: z_offs + rz, y_offs: y_offs + ry, x_offs: x_offs + rx] += slice_pred[0].sigmoid() * weight_template
            cell_flow[:, z_offs: z_offs + rz, y_offs: y_offs + ry, x_offs: x_offs + rx] += slice_pred[1:] * weight_template
            weight[z_offs: z_offs + rz, y_offs: y_offs + ry, x_offs: x_offs + rx] += weight_template
    weight += 1e-12
    cell_prob /= weight
    cell_flow /= weight
    cell_prob = cell_prob.cpu()[rz // 2: -rz // 2, ry // 2: -ry // 2, rx // 2: -rx // 2].contiguous().numpy()
    cell_flow = cell_flow.cpu()[:, rz // 2: -rz // 2, ry // 2: -ry // 2, rx // 2: -rx // 2].contiguous().numpy()
    np.save(os.path.join(args.output_dir, f"sig{sig_chn}_ref{ref_chn}_z{z0:04d}_y{y0:04d}_x{x0:04d}_cell_prob.npy"), cell_prob)
    np.save(os.path.join(args.output_dir, f"sig{sig_chn}_ref{ref_chn}_z{z0:04d}_y{y0:04d}_x{x0:04d}_cell_flow.npy"), cell_flow)

