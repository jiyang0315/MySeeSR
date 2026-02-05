#!/bin/bash
# ============================================
# 实验B: 固定权重版本（多尺度但权重不可学习）
# 目的：证明"可学习权重"相比"固定权重"的优势
# ============================================

CUDA_VISIBLE_DEVICES="2,3" accelerate launch --num_processes=2 --main_process_port 12347 train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/seesr_fixed_weights" \
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
--spatial_noise_debug_every 200 \
--use_multi_scale_conditioning \
--multi_scale_progressive \
--multi_scale_init_value 1.0 \
--multi_scale_edge_scales "1.0,0.5,0.25" \
--log_scale_weights_every 100

# 注意：有 --use_multi_scale_conditioning 和 --multi_scale_progressive
# 但是 **没有** --multi_scale_learnable
# 权重会使用渐进式初始化 [1.2, 1.1, 1.0, 0.9, 0.8] 但在训练中保持固定

