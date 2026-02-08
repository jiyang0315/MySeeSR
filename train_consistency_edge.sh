#!/bin/bash
# 实验B：+ Edge（只加边缘一致性损失）
# 目的：验证边缘约束的单独效果

CUDA_VISIBLE_DEVICES="2,3" accelerate launch --num_processes=2 --main_process_port 12345 train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/consistency_edge" \
--root_folders 'preset/datasets/train_datasets/training_for_dape' \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=1 \
--gradient_accumulation_steps=32 \
--null_text_ratio=0.5 \
--dataloader_num_workers=0 \
--checkpointing_steps=10000 \
--spatial_noise_alpha 0.6 \
--spatial_noise_edge_type sobel \
--spatial_noise_edge_blur 0 \
--spatial_noise_debug_every 200 \
--use_multi_scale_conditioning \
--multi_scale_learnable \
--multi_scale_progressive \
--multi_scale_init_value 1.0 \
--multi_scale_edge_scales "1.0,0.5,0.25" \
--log_scale_weights_every 100 \
--seed=42 \
--report_to=tensorboard \
--use_consistency_loss \
--consistency_use_edge \
--consistency_edge_weight 0.1 \
--consistency_edge_loss_type l1

# 只启用边缘损失，用于消融实验

