import os
from tqdm import tqdm
import glob
import numpy as np
import torch


if __name__ == "__main__":
    flist = glob.glob("/path/to/cellmask.npy")
    flist.sort()
    print(len(flist), flist[-1])
    D = len(flist)
    H, W = np.load(flist[0]).shape
    print("allocating cellmask")
    cellmask_3d = np.empty([D, H, W], dtype=np.uint32)
    print("stitching")
    cm1 = np.load(flist[0])
    cellmask_3d[0, :, :] = cm1
    next_cell_id = cm1.max() + 1
    prev_id_map = np.arange(cm1.max() + 1)
    cm1 = torch.tensor(cm1, device="cuda")
    for z in tqdm(range(1, D)):
        cm2 = np.load(flist[z])
        cm2_max = cm2.max()
        cm2 = torch.tensor(cm2, device="cuda")
        cm12 = ((cm1.long() << 16) + cm2.long()).to(torch.uint32)
        keys, values = torch.unique(cm12, return_counts=True)
        area_dict = {k: v for k, v in zip(keys.tolist(), values.tolist())}
        iou_dict = {k: {} for k in range(1, cm2_max + 1)}
        for k in area_dict:
            id2 = k & 0xffff
            id1 = k >> 16
            if id1 != 0 and id2 != 0:
                iou_dict[id2][id1] = area_dict[k] / (area_dict[k] + area_dict.get(id2, 0) + area_dict.get(id1 << 16, 0))
        id_map = np.zeros([cm2_max + 1], dtype=np.int64)
        blacklist = set()
        for i in range(1, cm2_max + 1):
            iou_and_id1_list = [(v, id1) for id1, v in iou_dict[i].items() if id1 not in blacklist]
            if len(iou_and_id1_list) == 0:
                id_map[i] = next_cell_id
                next_cell_id += 1
            else:
                max_iou, id1 = max(iou_and_id1_list)
                id_map[i] = prev_id_map[id1]
                blacklist.add(id1)
        id_map = torch.tensor(id_map, device="cuda")
        cm2_mapped = id_map[cm2.long()]
        cellmask_3d[z] = cm2_mapped.to(torch.uint32).cpu().numpy()
        cm1 = cm2
        prev_id_map = id_map
    np.save("cellmask_3d.npy", cellmask_3d)
