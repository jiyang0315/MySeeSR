import torch
import torch.nn.functional as F
from typing import List, Dict


def compute_edge_strength(lr_rgb: torch.Tensor, edge_type: str = "sobel", edge_blur: int = 0) -> torch.Tensor:
    """
    Compute a normalized edge strength map in [0, 1].

    Args:
        lr_rgb: (B, 3, H, W) tensor, expected in [0, 1] (but not strictly required).
        edge_type: "sobel" or "laplacian".
        edge_blur: Optional blur kernel size (0 disables). If even, it will be rounded up to the next odd value.

    Returns:
        edge_strength: (B, 1, H, W) tensor in [0, 1].
    """
    if lr_rgb.ndim != 4 or lr_rgb.shape[1] != 3:
        raise ValueError(f"lr_rgb must have shape (B, 3, H, W), got {tuple(lr_rgb.shape)}")

    x = lr_rgb
    if not x.is_floating_point():
        x = x.float()

    # Convert to grayscale for edge detection
    gray = 0.2989 * x[:, 0:1, ...] + 0.5870 * x[:, 1:2, ...] + 0.1140 * x[:, 2:3, ...]

    edge_type = (edge_type or "").lower()
    if edge_type == "sobel":
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-12)
    elif edge_type == "laplacian":
        k = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        edge = F.conv2d(gray, k, padding=1).abs()
    else:
        raise ValueError(f"Unknown edge_type={edge_type!r}, expected 'sobel' or 'laplacian'")

    if edge_blur and int(edge_blur) > 0:
        k = int(edge_blur)
        if k % 2 == 0:
            k += 1
        edge = F.avg_pool2d(edge, kernel_size=k, stride=1, padding=k // 2)

    # Per-image normalize to [0, 1]
    edge_min = edge.amin(dim=(2, 3), keepdim=True)
    edge_max = edge.amax(dim=(2, 3), keepdim=True)
    edge = (edge - edge_min) / (edge_max - edge_min + 1e-12)
    return edge.clamp(0.0, 1.0)


def compute_multi_scale_edges(
    lr_rgb: torch.Tensor,
    scales: List[float] = [1.0, 0.5, 0.25],
    edge_type: str = "sobel",
    edge_blur: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    计算多尺度边缘特征，用于多尺度条件注入。
    
    在不同尺度下检测边缘，以捕获不同粒度的结构信息：
    - 大尺度(1.0)：细节纹理和精细边缘
    - 中尺度(0.5)：主要物体轮廓
    - 小尺度(0.25)：全局结构和粗糙布局
    
    Args:
        lr_rgb: (B, 3, H, W) 输入图像，范围[0, 1]
        scales: 尺度因子列表，1.0表示原始分辨率
        edge_type: 边缘检测算子类型 ("sobel" 或 "laplacian")
        edge_blur: 可选的模糊核大小
    
    Returns:
        multi_scale_features: 字典包含：
            - 'edges_list': List[Tensor] 每个尺度的边缘图
            - 'edges_concat': (B, len(scales), H, W) 拼接的边缘特征
            - 'edges_pyramid': Dict[str, Tensor] 不同分辨率的边缘金字塔
    
    Example:
        >>> lr = torch.randn(4, 3, 512, 512)
        >>> features = compute_multi_scale_edges(lr, scales=[1.0, 0.5, 0.25])
        >>> print(features['edges_concat'].shape)  # (4, 3, 512, 512)
    """
    if lr_rgb.ndim != 4 or lr_rgb.shape[1] != 3:
        raise ValueError(f"lr_rgb must be (B, 3, H, W), got {tuple(lr_rgb.shape)}")
    
    B, C, H, W = lr_rgb.shape
    edges_list = []
    edges_pyramid = {}
    
    for i, scale in enumerate(scales):
        if scale == 1.0:
            scaled_img = lr_rgb
        else:
            # 下采样到对应尺度
            new_h = max(8, int(H * scale))  # 确保最小8像素
            new_w = max(8, int(W * scale))
            scaled_img = F.interpolate(
                lr_rgb,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )
        
        # 在当前尺度计算边缘
        edge = compute_edge_strength(scaled_img, edge_type=edge_type, edge_blur=edge_blur)
        
        # 保存金字塔（保持原始分辨率）
        edges_pyramid[f'scale_{scale}'] = edge
        
        # 上采样回原始分辨率用于拼接
        if scale != 1.0:
            edge_upsampled = F.interpolate(
                edge,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        else:
            edge_upsampled = edge
        
        edges_list.append(edge_upsampled)
    
    # 拼接所有尺度: (B, len(scales), H, W)
    edges_concat = torch.cat(edges_list, dim=1)
    
    return {
        'edges_list': edges_list,
        'edges_concat': edges_concat,
        'edges_pyramid': edges_pyramid,
    }


def prepare_multi_scale_conditions(
    lr_rgb: torch.Tensor,
    target_resolutions: List[int] = [64, 32, 16, 8],
    edge_type: str = "sobel",
    include_rgb: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    为UNet的不同层级准备多尺度条件输入。
    
    这个函数为ControlNet在UNet的每个下采样层级准备相应分辨率的条件特征。
    
    Args:
        lr_rgb: (B, 3, H, W) 低分辨率输入图像
        target_resolutions: UNet各层的latent分辨率（从高到低）
        edge_type: 边缘检测类型
        include_rgb: 是否包含RGB信息
    
    Returns:
        conditions: Dict[分辨率 -> (B, C, res, res) tensor]
                   如果include_rgb=True, C=4 (RGB+edge)
                   否则 C=1 (仅edge)
    
    Example:
        >>> lr = torch.randn(2, 3, 512, 512)
        >>> conds = prepare_multi_scale_conditions(lr, target_resolutions=[64, 32, 16, 8])
        >>> for res, feat in conds.items():
        ...     print(f"Resolution {res}: {feat.shape}")
        Resolution 64: torch.Size([2, 4, 64, 64])
        Resolution 32: torch.Size([2, 4, 32, 32])
        Resolution 16: torch.Size([2, 4, 16, 16])
        Resolution 8: torch.Size([2, 4, 8, 8])
    """
    conditions = {}
    
    for res in target_resolutions:
        # 下采样RGB到目标分辨率
        rgb_scaled = F.interpolate(
            lr_rgb,
            size=(res, res),
            mode='bilinear',
            align_corners=False
        )
        
        # 计算该尺度的边缘
        edge_scaled = compute_edge_strength(rgb_scaled, edge_type=edge_type)
        
        # 组合RGB和边缘特征
        if include_rgb:
            cond_feat = torch.cat([rgb_scaled, edge_scaled], dim=1)  # (B, 4, res, res)
        else:
            cond_feat = edge_scaled  # (B, 1, res, res)
        
        conditions[res] = cond_feat
    
    return conditions

