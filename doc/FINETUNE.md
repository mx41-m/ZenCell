## Finetune
You can download our pre-trained checkpoints from [Here](https://example.com), or pre-train the model on your own data by following the instructions in [PRETRAIN.md](PRETRAIN.md).

To fine-tune the model with multi-node distributed training, we provide an example script in `exps/finetune_3d.sh` or you could run the following on 1 nodes with 8 GPUs:

<pre> torchrun --nproc_per_node 8 main_finetune_3d.py \
    --model ${model} \
    --output_dir ${output_dir} \
    --log_dir ${output_dir} \
    --data_root /path/to/annotations \
    --finetune output_dir/${ft_exp_name}/checkpoint-${ft_epoch}.pth \
    --batch_size 2 --accum_iter 4 \
    --epochs 5000 --warmup_epochs 50 \
    --blr 1e-3 --layer_decay 0.85 --weight_decay 0.05 --drop_path 0.2 \
    2>&1 | tee ${output_dir}/output.log
</pre>
- `model`: The fine-tuning model shares the same backbone as the pre-trained model, but replaces the decoder with a segmentation head. In our case, we use `vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale`, which is defined in `models_vit.py`.
- `output_dir`: Directory where fine-tuning results are saved (e.g., model checkpoints and logs). 
- `data_root`: Path to the dataset directory for fine-tuning. The details of the dataset format are provided in the following [**Dataset Format**](#dataset-format) section. 
- `finetune`: Path to the pre-trained model checkpoint used to initialize the fine-tuned model.
- `batch_size`: Per-GPU batch size. In this example, it is set to 2.
- `accum_iter`: Number of iterations over which gradients are accumulated before an update step.
- `epochs`: Total number of fine-tuning epochs.
- `warmup_epochs`: Number of warmup epochs for the learning rate schedule.
- `blr`: Base learning rate.
- `layer_decay`: Layer-wise learning rate decay (following strategies from ELECTRA/BEiT).
- `weight_decay`: Weight decay (L2 regularization).
- `drop_path`:Drop path rate for the model (a regularization technique that randomly drops residual paths during training to improve generalization).

Additional hyperparameters and their detailed explanations can be found in `main_finetune_3d.py`.

### Dataset Format
For fine-tuning the pre-trained model, the input image format is the same as in pre-training: images are cropped from the complete samples, while the corresponding labels are loaded separately.

Inside the `data_root` folder, there should be a CSV file named `train.csv`. Each row corresponds to one training sample and contains the following attributes:
- `ID`: Integer identifier for each input image.
- `corner`: The upper-left corner of the input image, specified as (z,x,y). 
- `source`: Path to the complete sample from which the input image is cropped. The file format should be Zarr.
- `ref_channel`: Index of the reference channel stored in the complete Zarr sample.
- `channel`: Index of the signal channel stored in the complete Zarr sample.
- `crop_size`: The region of interest (ROI) size; the corresponding label size must match this.

All corresponding labels are saved in a subfolder named `train`, where each label file follows the format `{ID}_mask.npy`, with `{ID}` matching the `ID` value in the `train.csv` file.
