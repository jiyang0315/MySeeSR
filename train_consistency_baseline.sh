#!/bin/bash
# 实验A：Baseline（无一致性损失）
# 目的：作为对比基准
# 修改：与 train.sh 保持一致的配置，只是不加一致性损失

export MODEL_DIR="preset/models/stable-diffusion-2-base"
export OUTPUT_DIR="./experience/consistency_baseline"

CUDA_VISIBLE_DEVICES="4,5" accelerate launch --num_processes=2 --main_process_port 12345 train_seesr.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
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
 --multi_scale_learnable \
 --multi_scale_progressive \
 --multi_scale_init_value 1.0 \
 --multi_scale_edge_scales "1.0,0.5,0.25" \
 --log_scale_weights_every 100 \
 --seed=42 \
 --report_to=tensorboard

# 注意：不添加 --use_consistency_loss
# 这是 baseline，用于对比
# 所有其他配置与 train.sh 完全一致

