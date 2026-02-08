#!/bin/bash
# 实验D：+ Edge + Frequency（推荐配置）
# 修改：与 train.sh 保持一致的配置 + 一致性损失

export MODEL_DIR="preset/models/stable-diffusion-2-base"
export OUTPUT_DIR="./experience/consistency_edge_freq"

CUDA_VISIBLE_DEVICES="4,5" accelerate launch --num_processes=2 --main_process_port 12348 train_seesr.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
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
 --consistency_use_frequency \
 --consistency_edge_weight 0.1 \
 --consistency_frequency_weight 0.1 \
 --consistency_edge_loss_type l1 \
 --consistency_freq_loss_type l1 \
 --consistency_high_freq_weight 2.0

# 推荐配置：边缘 + 频域一致性损失（轻量级，无VGG）

