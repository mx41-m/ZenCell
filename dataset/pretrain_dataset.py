import torch
import zarr
import numpy as np
import os
import cv2
import tarfile


zarr_paths = [
]
zarr_cache_paths = [
]

class BrainCellPretrainDataset(torch.utils.data.Dataset):

    def __init__(self, patch_size, stride) -> None:
        self.patch_size = patch_size
        self.stride = stride

        self.vol_sizes = []
        self.sample_channels = []
        for zarr_path in zarr_paths:
            f = zarr.open(store=zarr.N5Store(zarr_path), mode="r")
            arr = f["setup3"]["timepoint0"]["s0"]
            assert len(arr.shape) == len(patch_size) == len(stride)
            self.vol_sizes.append(arr.shape)
            assert self._get_sample_len(arr.shape) > 0
            self.sample_channels.append([(3, 1), (3, 2), (3, 4)])
        self.zarr_files = None

        self.cache_mmaps = [{f"setup{x}": None for x in (1, 2, 3, 4)} for _ in (0, 1)]

    def _get_sample_len(self, vol_size):
        vol_len = 1
        for l, p, s in zip(vol_size, self.patch_size, self.stride):
            vol_len *= max((l - p) // s + 1, 0)
        return vol_len

    def __len__(self):
        total_len = 0
        for i in range(len(zarr_paths)):
            vol_len = self._get_sample_len(self.vol_sizes[i])
            total_len += vol_len * len(self.sample_channels[i])
        return total_len

    def __str__(self):
        return f"""BrainCellPretrainDataset(
    zarr_paths={zarr_paths},
    patch_size={self.patch_size},
    stride={self.stride},
    total_samples={len(self)},
)"""

    def __getitem__(self, idx: int) -> torch.Tensor:
        zarr_idx, chn_idx = 0, 0
        while zarr_idx < len(zarr_paths):
            vol_len = self._get_sample_len(self.vol_sizes[zarr_idx])
            sample_len = vol_len * len(self.sample_channels[zarr_idx])
            if idx >= sample_len:
                idx -= sample_len; zarr_idx += 1
            else:
                chn_idx = idx // vol_len
                idx -= chn_idx * vol_len
                break
        assert zarr_idx < len(zarr_paths)

        z_steps, y_steps, x_steps = [(l - p) // s + 1 for l, p, s in zip(
            self.vol_sizes[zarr_idx], self.patch_size, self.stride
        )]
        z_st = idx // (y_steps * x_steps) * self.stride[0]
        y_st = (idx % (y_steps * x_steps)) // x_steps * self.stride[1]
        x_st = idx % x_steps * self.stride[2]
        z_ps, y_ps, x_ps = self.patch_size
        z_ed, y_ed, x_ed = z_st + z_ps, y_st + y_ps, x_st + x_ps
        #print(idx, zarr_idx, chn_idx, z_st, z_ed, y_st, y_ed, x_st, x_ed)

        if self.zarr_files is None:
            self.zarr_files = [
                zarr.open(store=zarr.N5Store(zarr_path), mode="r")
                for zarr_path in zarr_paths
            ]

        arr_slice = []
        for chn in self.sample_channels[zarr_idx][chn_idx]:
            #arr = self.zarr_files[zarr_idx][f"setup{chn}"]["timepoint0"]["s0"]
            if self.cache_mmaps[zarr_idx][f"setup{chn}"] is None:
                self.cache_mmaps[zarr_idx][f"setup{chn}"] = np.memmap(
                    zarr_cache_paths[zarr_idx][f"setup{chn}"],
                    mode="r",
                    shape=self.zarr_files[zarr_idx][f"setup{chn}"]["timepoint0"]["s0"].shape,
                    dtype=np.uint16,
                )
            arr = self.cache_mmaps[zarr_idx][f"setup{chn}"]
            chn_slice = arr[z_st: z_ed, y_st: y_ed, x_st: x_ed]
            chn_slice = torch.from_numpy(chn_slice)
            #chn_slice = chn_slice.float()
            #chn_slice = (chn_slice - chn_slice.mean()) / (chn_slice.std() + 1e-6)
            arr_slice.append(chn_slice)
        arr_slice = torch.stack(arr_slice, axis=0)

        return arr_slice

