#!/usr/bin/env python

from cellpose.dynamics import compute_masks
import numpy as np
import glob
import os


rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NPROCS"])

file_list = []
for f in glob.glob("/path/to/blocks/cellmask/*_cell_prob.npy"):
    if os.path.isfile(f.replace("_cell_prob.npy", "_cellmask.npy")) and os.path.isfile(f.replace("_cell_prob.npy", "_cellp.npy")):
        if rank == 0:
            print("skip:", f)
        continue
    file_list.append(f)
file_list.sort()

if rank == 0:
    print(f"world_size={world_size}, num_tasks: {len(file_list)}")
    from tqdm import tqdm
else:
    tqdm = lambda x: x

for prob_path in tqdm(file_list[rank::world_size]):
    cell_prob = np.load(prob_path)
    cell_flow = np.load(prob_path.replace("_cell_prob.npy", "_cell_flow.npy"))
    cellmask, cellp = compute_masks(cell_flow, cell_prob, min_size=0, flow_threshold=None, cellprob_threshold=.5, do_3D=True)
    np.save(prob_path.replace("_cell_prob.npy", "_cellmask.npy"), cellmask)
    np.save(prob_path.replace("_cell_prob.npy", "_cellp.npy"), cellp)
