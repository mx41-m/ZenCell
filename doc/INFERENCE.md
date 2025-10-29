## Inference for whole brain
To inference the 3D cell segmentation for whole brain, we provide an example script in `exps/inference_wholebrain.sh` or you should run the followings on 1 nodes with 8 GPUs:

<pre> torchrun --nproc_per_node 8 python -u inference_wholebrain_3d.py \
     --model vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale \
     --ckpt  \ # path to finetune model checkpoint
     --chns  \ # lists of reference channel and signal channel, e.g., 3,0 3,1 3,2
     --output_dir  /path/to/blocks/cellmask/ \ # output save directory
     --zarr_path # original raw brain images path
</pre>
- `model`: The fine-tuned model architecture. In our case, we use `vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale`, which is defined in `models_vit.py`.
- `ckpt`: The checkpoint file path for the fine-tuned model.
- `chns`: A list of all channels required for generating whole-brain 3D cell segmentation. Each channel pair follows the format `reference_channel,signal_channel`, with different pairs separated by spaces. For example, `3,0 3,1 3,2` indicates that whole-brain 3D segmentation will be generated for signal channels 0, 1, and 2, using channel 3 as the reference channel.
- `output_dir`: The directory where the output results will be saved.
- `zarr_path`: The file path to the original raw brain image data.

Considering that whole-brain datasets are typically very large, our script `inference_wholebrain_3d.py` automatically divides the brain volume into multiple chunks, each of size `640, 1024, 1024` in `(z,y,x)` order. The script then performs 3D cell instance segmentation on each chunk independently. Afterward, users can run `compute_masks_all_3d.py` to generate 3D cell instance masks for all chunks. In this script, users need to specify the path to `/path/to/blocks/cellmask/`, which corresponds to the `output_dir` from `inference_wholebrain_3d.py` (see line 13 in `compute_masks_all_3d.py`).

After inference, users can merge all segmented chunks into a single, whole-brain 3D cell instance segmentation using `whole_samples_stitch_util/stitch_3d.py`. In this script, users must specify: 
- The path to raw whole samples in line 61
- The reference and signal channels in line 63, 64, and 
- The path to the directory containing the segmented chunks (i.e., the `output_dir`, `/path/to/blocks/cellmask/`, from `inference_wholebrain_3d.py`) in line 112, 125, and 126, as well as the output path for saving the final stitched whole-brain 3D segmentation results in line 68.
