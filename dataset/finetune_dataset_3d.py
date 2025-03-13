import torch
import pandas as pd
import ast
import multiprocessing.managers
import torch.distributed as dist
import numpy as np
from cellpose.dynamics import masks_to_flows
from typing import Tuple
import os
import random


class FinetuneDataset3D(torch.utils.data.Dataset):

    def __init__(self, data_root: str, split: str, roi_size: Tuple[int, int, int],
                 ctx_size: Tuple[int, int, int], alignment: Tuple[int, int, int]) -> None:
        df = pd.read_csv(os.path.join(data_root, split + ".csv"))
        self.ctx_size = ctx_size
        self.roi_size = roi_size
        
        print("Preloading 3D training data")
        self.sample_sizes = [ast.literal_eval(x) for x in df["crop_size"].tolist()]
        cz, cy, cx = ctx_size
        rz, ry, rx = roi_size
        az, ay, ax = alignment
        assert cz >= rz and cy >= ry and cx >= rx
        self.roi_offs = [(cz - rz) // 2 // az * az, (cy - ry) // 2 // ay * ay, (cx - rx) // 2 // ax * ax]
        shm_size = 0
        self.shm_offs = []
        for z, y, x in self.sample_sizes:
            assert z >= rz and y >= ry and x >= rx
            self.shm_offs.append(shm_size)
            shm_size += (z + cz - rz) * (y + cy - ry) * (x + cx - rx) * 4 + z * y * x * 2  # input: 2channel * uint16, label: uint16
        smm = multiprocessing.managers.SharedMemoryManager()
        smm.start()
        self.shm = smm.SharedMemory(size=shm_size)
        for i in range(len(df)):
            z1, y1, x1 = ast.literal_eval(df.iloc[i]["corner"])
            z0, y0, x0 = z1 - self.roi_offs[0], y1 - self.roi_offs[1], x1 - self.roi_offs[2]
            z, y, x = self.sample_sizes[i]
            source = df.iloc[i]["source"]
            ref_chn = df.iloc[i]["ref_channel"]
            sig_chn = df.iloc[i]["channel"]
            ref_arr, sig_arr = self._read_channels_from_zarr(
                source, [ref_chn, sig_chn], [z0, y0, x0], [z + cz - rz, y + cy - ry, x + cx - rx],
            )
            label_path = os.path.join(data_root, split, f"{df.iloc[i]['ID']:04d}_mask.npy")
            label_arr = np.load(label_path)
            ref_arr_shm, sig_arr_shm, label_arr_shm = self._np_array_from_shm(i)
            ref_arr_shm[...] = ref_arr
            sig_arr_shm[...] = sig_arr
            label_arr_shm[...] = label_arr

    def _read_channels_from_zarr(self, source, channels, corner, size):
        import zarr
        ret = []
        arr_slices = tuple(slice(max(x, 0), x + y) for x, y in zip(corner, size))
        arr_shape = None

        zarr_path = f"/path/to/fused.n5"
        zarr_fp = zarr.open(zarr_path, mode="r")
        arr_shape = zarr_fp["setup0"]["timepoint0"]["s0"].shape
        for chn in channels:
            ret.append(zarr_fp[f"setup{chn}"]["timepoint0"]["s0"][arr_slices])

        assert len(ret) == len(channels)
        assert arr_shape is not None and len(arr_shape) == 3
        pad_arr = tuple((max(-x, 0), max(x + y - z, 0)) for x, y, z in zip(corner, size, arr_shape))
        if any(x > 0 or y > 0 for x, y in pad_arr):
            ret = [np.pad(x, pad_arr) for x in ret]
        assert all(x.shape == tuple(size) for x in ret)
        return ret

    def _np_array_from_shm(self, idx):
        z, y, x = self.sample_sizes[idx]
        cz, cy, cx = self.ctx_size
        rz, ry, rx = self.roi_size
        shm_offs_ref = self.shm_offs[idx]
        shm_offs_sig = shm_offs_ref + (z + cz - rz) * (y + cy - ry) * (x + cx - rx) * 2
        shm_offs_label = shm_offs_sig + (z + cz - rz) * (y + cy - ry) * (x + cx - rx) * 2

        ref_arr = np.ndarray([z + cz - rz, y + cy - ry, x + cx - rx], dtype=np.uint16, buffer=self.shm.buf, offset=shm_offs_ref)
        sig_arr = np.ndarray([z + cz - rz, y + cy - ry, x + cx - rx], dtype=np.uint16, buffer=self.shm.buf, offset=shm_offs_sig)
        label_arr = np.ndarray([z, y, x], dtype=np.uint16, buffer=self.shm.buf, offset=shm_offs_label)
        
        return ref_arr, sig_arr, label_arr
 
    def __len__(self):
        return len(self.shm_offs)
 
    def __getitem__(self, idx: int):
        z, y, x = self.sample_sizes[idx]
        rz, ry, rx = self.roi_size
        ref_arr, sig_arr, label_arr = self._np_array_from_shm(idx)
        z0, y0, x0 = random.randrange(z - rz + 1), random.randrange(y - ry + 1), random.randrange(x - rx + 1)
        cz, cy, cx = self.ctx_size

        input_vol = np.stack([ref_arr[z0: z0 + cz, y0: y0 + cy, x0: x0 + cx], sig_arr[z0: z0 + cz, y0: y0 + cy, x0: x0 + cx]], axis=0)
        input_vol = torch.from_numpy(input_vol)
        input_vol = input_vol.float()
        input_vol = input_vol - input_vol.mean(dim=(1, 2, 3), keepdim=True)
        input_vol = input_vol / (input_vol.std(dim=(1, 2, 3), keepdim=True) + 1e-6)

        label_vol = label_arr[z0: z0 + rz, y0: y0 + ry, x0: x0 + rx]
        label_bin = label_vol >= 1
        label_bin = torch.from_numpy(label_bin).float()

        label_flow = masks_to_flows(label_vol)
        
        return input_vol, label_bin, label_flow

