'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys
sys.path.append(os.getcwd())
import cv2
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

import argparse
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from ram.models import ram
from ram import inference_ram as inference

parser = argparse.ArgumentParser()
parser.add_argument("--gt_path", nargs='+', default=['PATH 1', 'PATH 2'], help='the path of high-resolution images')
parser.add_argument("--save_dir", type=str, default='preset/datasets/train_datasets/training_for_seesr', help='the save path of the training dataset.')
parser.add_argument("--start_gpu", type=int, default=1, help='if you have 5 GPUs, you can set it to 1/2/3/4/5 on five gpus for parallel processing., which will save your time. ')  
parser.add_argument("--batch_size", type=int, default=10, help='smaller batch size means much time but more extensive degradation for making the training dataset.')  
parser.add_argument("--epoch", type=int, default=1, help='decide how many epochs to create for the dataset.')
args = parser.parse_args()

print(f'====== START GPU: {args.start_gpu} =========')
seed_everything(24+args.start_gpu*1000)

from torchvision.transforms import Normalize, Compose
args_training_dataset = {}

# Please set your gt path here. If you have multi dirs, you can set it as ['PATH1', 'PATH2', 'PATH3', ...]
args_training_dataset['gt_path'] = args.gt_path

#################### REALESRGAN SETTING ###########################
args_training_dataset['queue_size'] = 160
args_training_dataset['crop_size'] =  512
args_training_dataset['io_backend'] = {}
args_training_dataset['io_backend']['type'] = 'disk'

args_training_dataset['blur_kernel_size'] = 21
args_training_dataset['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_training_dataset['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
args_training_dataset['sinc_prob'] = 0.1
args_training_dataset['blur_sigma'] = [0.2, 3]
args_training_dataset['betag_range'] = [0.5, 4]
args_training_dataset['betap_range'] = [1, 2]

args_training_dataset['blur_kernel_size2'] = 11
args_training_dataset['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_training_dataset['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
args_training_dataset['sinc_prob2'] = 0.1
args_training_dataset['blur_sigma2'] = [0.2, 1.5]
args_training_dataset['betag_range2'] = [0.5, 4.0]
args_training_dataset['betap_range2'] = [1, 2]

args_training_dataset['final_sinc_prob'] = 0.8

args_training_dataset['use_hflip'] = True
args_training_dataset['use_rot'] = False

train_dataset = RealESRGANDataset(args_training_dataset)
batch_size = args.batch_size
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=11,
    drop_last=True,
)

#################### SIMPLE DEGRADATION SETTING ###########################
args_degradation = {}
# Simple single-stage degradation
args_degradation['blur_kernel_size'] = 21
args_degradation['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_degradation['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
args_degradation['blur_sigma'] = [0.2, 3]
args_degradation['downsample_range'] = [2.0, 4.0]  # scale factor range for downsampling
args_degradation['noise_range'] = [0, 15]  # gaussian noise sigma range, None to disable
args_degradation['jpeg_range'] = [60, 95]  # jpeg quality range, None to disable

args_degradation['gt_size'] = 512


from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
import torch.nn.functional as F

def simple_degradation(batch, args_degradation):
    """
    Simple single-stage degradation pipeline:
    1. blur (mixed kernels)
    2. downsample
    3. add gaussian noise (optional)
    4. jpeg compression (optional)
    5. resize back to original size
    """
    im_gt = batch['gt'].cuda()
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    
    # Get batch size and original size
    b, c, h, w = im_gt.shape
    
    # Convert tensor to numpy for degradation operations (per image in batch)
    lq_batch = []
    gt_batch = []
    
    for i in range(b):
        # Get single image and convert to numpy [0, 1]
        img_gt = im_gt[i].permute(1, 2, 0).cpu().numpy()  # (h, w, c), RGB, [0, 1]
        
        # Convert RGB to BGR for cv2 operations
        img_gt = img_gt[..., ::-1]  # BGR
        
        # ------------------------ generate lq image ------------------------ #
        # 1. blur
        kernel = random_mixed_kernels(
            args_degradation['kernel_list'],
            args_degradation['kernel_prob'],
            args_degradation['blur_kernel_size'],
            args_degradation['blur_sigma'],
            args_degradation['blur_sigma'],
            [-math.pi, math.pi],
            noise_range=None,
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        
        # 2. downsample
        scale = np.random.uniform(args_degradation['downsample_range'][0], args_degradation['downsample_range'][1])
        img_lq = cv2.resize(
            img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR
        )
        
        # 3. add gaussian noise
        if args_degradation['noise_range'] is not None:
            img_lq = random_add_gaussian_noise(img_lq, args_degradation['noise_range'])
        
        # 4. jpeg compression
        if args_degradation['jpeg_range'] is not None:
            img_lq = random_add_jpg_compression(img_lq, args_degradation['jpeg_range'])
        
        # 5. resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert back: BGR to RGB, [0, 1]
        img_lq = img_lq[..., ::-1].astype(np.float32)
        img_gt = img_gt[..., ::-1].astype(np.float32)
        
        # Convert to tensor
        lq_tensor = torch.from_numpy(img_lq).permute(2, 0, 1).unsqueeze(0)  # (1, c, h, w)
        gt_tensor = torch.from_numpy(img_gt).permute(2, 0, 1).unsqueeze(0)  # (1, c, h, w)
        
        lq_batch.append(lq_tensor)
        gt_batch.append(gt_tensor)
    
    # Stack all images in batch
    lq = torch.cat(lq_batch, dim=0).cuda()
    gt = torch.cat(gt_batch, dim=0).cuda()
    
    # Random crop to gt_size
    gt_size = args_degradation['gt_size']
    gt, lq = paired_random_crop(gt, lq, gt_size, scale=1)
    
    # Clamp to [0, 1]
    gt = torch.clamp(gt, 0, 1)
    lq = torch.clamp(lq, 0, 1)
    
    return lq, gt


root_path = args.save_dir
gt_path = os.path.join(root_path, 'gt')
lr_path = os.path.join(root_path, 'lr')
sr_bicubic_path = os.path.join(root_path, 'sr_bicubic')
os.makedirs(gt_path, exist_ok=True)
os.makedirs(lr_path, exist_ok=True)
os.makedirs(sr_bicubic_path, exist_ok=True)


epochs = args.epoch
step = len(train_dataset) * epochs * args.start_gpu
with torch.no_grad():
    for epoch in range(epochs):
        for num_batch, batch in enumerate(train_dataloader):
            lr_batch, gt_batch = simple_degradation(batch, args_degradation=args_degradation)
            sr_bicubic_batch = F.interpolate(lr_batch, size=(gt_batch.size(-2), gt_batch.size(-1)), mode='bicubic',)

            for i in range(batch_size):
                step += 1
                print('process {} images...'.format(step))
                lr = lr_batch[i, ...]
                gt = gt_batch[i, ...]
                sr_bicubic = sr_bicubic_batch[i, ...]

                lr_save_path =  os.path.join(lr_path,'{}.png'.format(str(step).zfill(7)))
                gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))
                sr_bicubic_save_path =  os.path.join(sr_bicubic_path, '{}.png'.format(str(step).zfill(7)))

                cv2.imwrite(lr_save_path, 255*lr.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
                cv2.imwrite(gt_save_path, 255*gt.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
                cv2.imwrite(sr_bicubic_save_path, 255*sr_bicubic.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
               

            del lr_batch, gt_batch, sr_bicubic_batch
            torch.cuda.empty_cache()
    