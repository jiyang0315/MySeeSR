CUDA_VISIBLE_DEVICES=1 python test_seesr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /home/jiyang/jiyang/Projects/SeeSR/experience/consistency_full/checkpoint-30000 \
--image_path /home/jiyang/jiyang/Projects/SeeSR/preset/datasets/test_datasets/DIV2K/LR \
--output_dir preset/datasets/seesr_multi_scale_full_30000 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 