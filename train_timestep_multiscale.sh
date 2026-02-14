#!/bin/bash
# ============================================================================
# 实验C: 时间步自适应 + 多尺度条件注入（推荐配置）
# 用于对比：展示两种方法的协同效果
# ============================================================================

CUDA_VISIBLE_DEVICES="4,5" accelerate launch --num_processes=2 --main_process_port 12346 train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/timestep_multiscale" \
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
--spatial_noise_debug_every 200 \
--use_multi_scale_conditioning \
--multi_scale_learnable \
--multi_scale_progressive \
--multi_scale_init_value 1.0 \
--multi_scale_edge_scales "1.0,0.5,0.25" \
--log_scale_weights_every 100 \
--use_timestep_adaptive \
--timestep_strategy cosine \
--timestep_max_weight 1.3 \
--timestep_min_weight 0.7 \
--timestep_learnable \
--timestep_combination multiply \
--log_timestep_weights_every 500
# 注意：启用多尺度 + 时间步自适应，这是推荐的配置

