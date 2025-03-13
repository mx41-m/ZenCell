ft_epoch=3
ft_exp_name=mae_vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale_10ep_mr0.90_d40_256px
model=vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale
output_dir=output_dir/${ft_exp_name}/ft3d_ep${ft_epoch}_74vols_v7_5000ep

mkdir -p ${output_dir}

export LD_PRELOAD=/path/to/gperftools-2.16/lib/libtcmalloc.so

torchrun --nproc_per_node 8 main_finetune_3d.py \
    --model ${model} \
    --output_dir ${output_dir} \
    --log_dir ${output_dir} \
    --data_root /path/to/annotations --data_split train \
    --finetune output_dir/${ft_exp_name}/checkpoint-${ft_epoch}.pth \
    --batch_size 2 --accum_iter 4 \
    --epochs 5000 --warmup_epochs 50 \
    --blr 1e-3 --layer_decay 0.85 --weight_decay 0.05 --drop_path 0.2 \
    2>&1 | tee ${output_dir}/output.log
