#!/bin/bash
# ============================================
# 实验A: Baseline（无多尺度条件注入）
# 目的：作为基准，证明多尺度方法的有效性
# ============================================

CUDA_VISIBLE_DEVICES="6,7" accelerate launch --num_processes=2 --main_process_port 12348 train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/seesr_baseline" \
--root_folders 'preset/datasets/train_datasets/training_for_dape' \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=16 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=10000 \
--spatial_noise_alpha 0.6 \
--spatial_noise_edge_type sobel \
--spatial_noise_edge_blur 0 \
--spatial_noise_debug_every 200

# 注意：移除了所有 --use_multi_scale_conditioning 相关参数
# 这是标准的SeeSR训练，没有多尺度条件注入

