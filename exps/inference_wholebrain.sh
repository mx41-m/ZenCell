#!/usr/bin/env sh

torchrun --nproc_per_node 8 python -u inference_wholebrain_3d.py \
    --model vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale \
    --ckpt  \ # path to finetune model checkpoint
    --chns  \ # lists of reference channel and signal channel, e.g., 3,0 3,1 3,2
    --output_dir  \ # output save directory
    --zarr_path # original raw brain images path

