#!/bin/bash
# ============================================================================
# 实验A: Baseline（无时间步自适应）
# 用于对比：展示时间步自适应的效果
# ============================================================================

CUDA_VISIBLE_DEVICES="4,5" accelerate launch --num_processes=2 --main_process_port 12345 train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/timestep_baseline" \
--root_folders 'preset/datasets/train_datasets/training_for_dape' \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=16 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=10000 \
--max_train_steps=100000 \
--spatial_noise_alpha 0.6 \
--spatial_noise_edge_type sobel \
--spatial_noise_edge_blur 0 \
--spatial_noise_debug_every 200
# 注意：不启用任何创新模块，作为baseline

