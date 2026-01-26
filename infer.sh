CUDA_VISIBLE_DEVICES=7 python test_seesr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path /home/jiyang/jiyang/Projects/SeeSR/experience/seesr_noise/checkpoint-100000 \
--image_path /home/jiyang/jiyang/Projects/SeeSR/preset/datasets/test_datasets/DIV2K/LR \
--output_dir preset/datasets/seesr_no_dape_noise_100000 \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 