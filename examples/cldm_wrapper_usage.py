"""
ControlLDM 封装类的使用示例
展示如何使用改进后的 CLDM 封装类
"""
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models import CLIPTextModel
from transformers import CLIPTokenizer

from models.cldm_wrapper import ControlLDM
from models.cldm import CLDMModel


def create_controllm(
    pretrained_model_path: str,
    seesr_model_path: str = None,
) -> ControlLDM:
    """
    创建 ControlLDM 实例
    
    Args:
        pretrained_model_path: 预训练 SD 模型路径
        seesr_model_path: SeeSR 模型路径（如果使用自定义模型）
    
    Returns:
        ControlLDM 实例
    """
    # 加载基础模型
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    
    if seesr_model_path:
        unet = UNet2DConditionModel.from_pretrained(seesr_model_path, subfolder="unet")
        controlnet = CLDMModel.from_pretrained(seesr_model_path, subfolder="controlnet")
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        # 从 UNet 初始化 ControlNet
        controlnet = CLDMModel.from_unet(unet, use_image_cross_attention=True)
    
    # 创建 ControlLDM
    controllm = ControlLDM(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        controlnet=controlnet,
    )
    
    return controllm


def example_training_setup():
    """训练设置示例"""
    controllm = create_controllm("path/to/pretrained_model")
    
    # 冻结基础模型
    controllm.freeze_base_models()
    
    # 解冻 ControlNet
    controllm.unfreeze_controlnet()
    
    # 启用梯度检查点以节省内存
    controllm.enable_gradient_checkpointing()
    
    # 启用 xformers（如果可用）
    controllm.enable_xformers()
    
    # 设置控制尺度
    controllm.set_control_scales([1.0] * 13)
    
    return controllm


def example_inference():
    """推理示例"""
    controllm = create_controllm("path/to/pretrained_model", "path/to/seesr_model")
    
    # 设置为评估模式
    controllm.eval()
    
    # 转换到半精度以节省内存
    controllm.cast_dtype(torch.float16)
    
    # 准备输入
    cond_img = torch.randn(1, 3, 512, 512)  # 条件图像
    prompt = "a beautiful landscape"
    
    # 准备条件
    cond = controllm.prepare_condition(
        cond_img=cond_img,
        txt=prompt,
        tiled=True,  # 使用 tiled VAE
        tile_size=256,
    )
    
    # 创建噪声
    scheduler = DDPMScheduler.from_pretrained("path/to/pretrained_model", subfolder="scheduler")
    latents = torch.randn(1, 4, 64, 64)
    timesteps = torch.tensor([500])
    
    # 前向传播
    with torch.no_grad():
        noise_pred = controllm(
            x_noisy=latents,
            t=timesteps,
            cond=cond,
        )
    
    return noise_pred


def example_from_unet_init():
    """从 UNet 初始化 ControlNet 的示例"""
    controllm = create_controllm("path/to/pretrained_model")
    
    # 从 UNet 初始化 ControlNet 权重
    init_with_new_zero, init_with_scratch = controllm.load_controlnet_from_unet()
    
    print(f"Keys initialized with new zeros: {len(init_with_new_zero)}")
    print(f"Keys initialized from scratch: {len(init_with_scratch)}")
    
    return controllm


if __name__ == "__main__":
    # 训练示例
    print("=== Training Setup ===")
    train_model = example_training_setup()
    
    # 推理示例
    print("\n=== Inference Example ===")
    # noise_pred = example_inference()
    
    # 从 UNet 初始化示例
    print("\n=== From UNet Init ===")
    # init_model = example_from_unet_init()






