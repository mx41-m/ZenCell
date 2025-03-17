import numpy as np
import zarr

import dask.array as da

zarr_fp = zarr.open("/path/to/raw/samples", mode="r")['0']
dask_arr = da.from_zarr(zarr_fp)
ref = # reference channel 
sig = # signal channel
D, H, W = dask_arr[sig].shape

save_file = zarr.group(store=zarr.N5Store('/path/to/save/results/sig'+str(sig)+'_ref'+str(ref)+'/whole_brain_mask_fused.n5'))
shape = (D, H, W)
chunks = (256, 256, 256)
if 'data' not in save_file:
    wholebrain = save_file.create_dataset(
            'data',
            shape=shape,
            chunks=chunks,
            dtype='int32',
            #compressor=zarr.Blosc(cname='zstd', clevel=3)
            )
else:
    wholebrain = save_file['data']


for z in range(0, D, 1000):
    for y in range(0, H, 1024):
        for x in range(0, W, 1024):
            z_sz = min(1000, D - z)
            y_sz = min(1000, H - y)
            x_sz = min(1000, W - x)
            print('/path/to/blocks/cellmask/sig'+str(sig)+'_ref'+str(ref)+'_z'+f'{z:04}_y'+f'{y:04}_x'+f'{x:04}'+'_cellmask.npy')
            wholebrain[z:z+z_sz, y:y+y_sz, x:x+x_sz] = np.load('/path/to/blocks/cellmask/sig'+str(sig)+'_ref'+str(ref)+'_z'+f'{z:04}_y'+f'{y:04}_x'+f'{x:04}'+'_cellmask.npy')[:z_sz,:y_sz,:x_sz]


