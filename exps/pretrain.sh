#!/usr/bin/env sh

model=mae_vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale
epochs=10
mask_ratio=0.90

exp_name=${model}_${epochs}ep_mr${mask_ratio}_d40_256px
output_dir=output_dir/${exp_name}

mkdir -p ${output_dir}
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1

torchrun --nproc_per_node 8 main_pretrain.py \
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
