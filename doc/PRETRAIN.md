## Pretrain
We recommend pre-training the model on your own dataset for optimal segmentation performance. However, we also provide our pre-trained checkpoints [Here](https://example.com) and details of our pre-training dataset are described in our [Paper](https://example.com). 

To pretrain the model with multi-node distributed training, we provide an example script in `exps/pretrain.sh` or you could run the following on 1 nodes with 8 GPUs:

<pre> torchrun --nproc_per_node 8 main_pretrain.py \
    --batch_size 4 --accum_iter 24 \
    --input_size 40 1280 1280 \
    --stride 20 128 128 \
    --model ${model} \
    --norm_pix_loss \
    --mask_ratio ${mask_ratio} \
    --epochs ${epochs} \
    --warmup_epochs 1 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --output_dir ${output_dir} --log_dir ${output_dir} \
    2>&1 | tee -a ${output_dir}/output.log
</pre>
- `batch_size`: Per-GPU batch size. In this example, it is set to 2.
- `accum_iter`: Number of iterations over which gradients are accumulated before an update step.
- `input_size`: The input size for the model, which includes both the region of interest (ROI) and the context. In our setting, the ROI size is [40, 256, 256] (z, x, y) and the context size is [40, 1280, 1280] (z, x, y). Detailed explanations of ROI and context can be found in our [Paper](https://example.com). 
- `stride`: The stride used when the model pre-trains across the entire sample (e.g., a whole mouse brain).
- `model`: The pre-training model. In our case, we use `mae_vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale`, which is defined in `models_mae.py`. 
- `norm_pix_loss`: Use (per-patch) normalized pixels as targets for computing loss.
- `mask_ratio`: The percentage of patches removed during pre-training. Detailed explanations of this parameter can be found in [Paper](https://example.com).
- `epochs`: Total number of fine-tuning epochs.
- `warmup_epochs`: Number of warmup epochs for the learning rate schedule.
- `blr`: Base learning rate.
- `weight_decay`: Weight decay (L2 regularization).
- `output_dir`: Directory where the pre-trained model checkpoints are saved.

Additional hyperparameters and their detailed explanations can be found in `main_pretrain.py`.

### Dataset Format
The input images for our pre-training model include both the region of interest (ROI) and the surrounding context. To support flexible context sizes and improve data-loading efficiency, we cut and load patches directly from the complete sample, rather than manually cropping the dataset beforehand.

The dataset format for pre-training can be flexible; in our case, we use Zarr as an example. To pretrain the model with new samples, provide the paths of all complete samples in the variable `zarr_paths` inside `dataset/pretrain_dataset.py`.
