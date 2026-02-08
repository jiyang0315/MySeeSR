"""
Multi-Task Consistency Losses for SeeSR
Implements edge, frequency, and perceptual consistency constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import torchvision.models as models


class EdgeConsistencyLoss(nn.Module):
    """
    边缘一致性损失：约束SR输出的边缘与HR的边缘一致
    使用多尺度Sobel算子提取边缘特征
    """
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        
        # Sobel算子 (用于边缘检测)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # 扩展到3通道
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算边缘强度
        Args:
            x: (B, C, H, W) 图像张量，范围[-1, 1]或[0, 1]
        Returns:
            edges: (B, C, H, W) 边缘强度图
        """
        # 对每个通道独立计算边缘
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        
        # 边缘强度 = sqrt(grad_x^2 + grad_y^2)
        edges = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        return edges
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) 预测图像
            target: (B, C, H, W) 目标图像
        Returns:
            loss: 标量损失
        """
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        
        if self.loss_type == "l1":
            loss = F.l1_loss(pred_edges, target_edges)
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred_edges, target_edges)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class FrequencyConsistencyLoss(nn.Module):
    """
    频域一致性损失：约束SR输出的频谱与HR的频谱一致
    使用FFT提取频域特征，主要关注高频成分
    """
    def __init__(self, loss_type: str = "l1", high_freq_weight: float = 2.0):
        super().__init__()
        self.loss_type = loss_type
        self.high_freq_weight = high_freq_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) 预测图像
            target: (B, C, H, W) 目标图像
        Returns:
            loss: 标量损失
        """
        # FFT变换到频域
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        
        # 幅度谱
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # 可选：对高频加权（边缘信息主要在高频）
        if self.high_freq_weight > 1.0:
            B, C, H, W = pred_amp.shape
            # 创建频率权重图（中心低频，边缘高频）
            freq_weight = torch.ones_like(pred_amp)
            center_h, center_w = H // 2, W // 2
            y_coords = torch.arange(H, device=pred.device).view(-1, 1).expand(H, W)
            x_coords = torch.arange(W, device=pred.device).view(1, -1).expand(H, W)
            
            # 距离中心的归一化距离
            dist = torch.sqrt(((y_coords - center_h) / center_h).pow(2) + 
                            ((x_coords - center_w) / center_w).pow(2))
            dist = dist.clamp(0, 1)
            
            # 高频加权：距离越远权重越大
            freq_weight = 1.0 + (self.high_freq_weight - 1.0) * dist
            freq_weight = freq_weight.view(1, 1, H, W).expand_as(pred_amp)
            
            pred_amp = pred_amp * freq_weight
            target_amp = target_amp * freq_weight
        
        # 计算损失
        if self.loss_type == "l1":
            loss = F.l1_loss(pred_amp, target_amp)
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred_amp, target_amp)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class PerceptualConsistencyLoss(nn.Module):
    """
    感知一致性损失：使用预训练VGG网络提取特征，约束感知相似度
    """
    def __init__(
        self, 
        loss_type: str = "l1",
        feature_layers: list = None,
        use_relu: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        
        # 默认使用VGG16的多层特征
        if feature_layers is None:
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.feature_layers = feature_layers
        
        # 加载预训练VGG16
        vgg = models.vgg16(pretrained=True).features.eval()
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        # 构建特征提取器
        self.feature_extractor = vgg
        
        # VGG层名到索引的映射
        self.layer_name_mapping = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
            'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29,
        }
        
        # VGG归一化参数（ImageNet统计量）
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入图像归一化到VGG的输入范围
        Args:
            x: (B, C, H, W) 图像，假设范围为[-1, 1]
        Returns:
            normalized: 归一化后的图像
        """
        # 转换到 [0, 1]
        x = (x + 1.0) / 2.0
        
        # 应用ImageNet归一化
        x = (x - self.mean) / self.std
        return x
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取多层VGG特征
        Args:
            x: (B, C, H, W) 归一化后的图像
        Returns:
            features: {layer_name: feature_tensor}
        """
        features = {}
        h = x
        for i, layer in enumerate(self.feature_extractor):
            h = layer(h)
            # 检查是否是目标层
            for layer_name, layer_idx in self.layer_name_mapping.items():
                if i == layer_idx and layer_name in self.feature_layers:
                    features[layer_name] = h
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) 预测图像，范围[-1, 1]
            target: (B, C, H, W) 目标图像，范围[-1, 1]
        Returns:
            loss: 标量损失
        """
        # 归一化输入
        pred_norm = self.normalize_input(pred)
        target_norm = self.normalize_input(target)
        
        # 提取特征
        pred_features = self.extract_features(pred_norm)
        target_features = self.extract_features(target_norm)
        
        # 计算多层特征损失
        loss = 0.0
        for layer_name in self.feature_layers:
            if layer_name in pred_features and layer_name in target_features:
                pred_feat = pred_features[layer_name]
                target_feat = target_features[layer_name]
                
                if self.loss_type == "l1":
                    loss += F.l1_loss(pred_feat, target_feat)
                elif self.loss_type == "l2":
                    loss += F.mse_loss(pred_feat, target_feat)
        
        # 平均损失
        loss = loss / len(self.feature_layers)
        
        return loss


class ConsistencyLossManager(nn.Module):
    """
    一致性损失管理器：统一管理多个一致性损失
    """
    def __init__(
        self,
        use_edge: bool = True,
        use_frequency: bool = True,
        use_perceptual: bool = False,
        edge_weight: float = 1.0,
        frequency_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        edge_loss_type: str = "l1",
        freq_loss_type: str = "l1",
        perceptual_loss_type: str = "l1",
        high_freq_weight: float = 2.0,
    ):
        super().__init__()
        
        self.use_edge = use_edge
        self.use_frequency = use_frequency
        self.use_perceptual = use_perceptual
        
        self.edge_weight = edge_weight
        self.frequency_weight = frequency_weight
        self.perceptual_weight = perceptual_weight
        
        # 初始化各损失模块
        if self.use_edge:
            self.edge_loss = EdgeConsistencyLoss(loss_type=edge_loss_type)
        
        if self.use_frequency:
            self.freq_loss = FrequencyConsistencyLoss(
                loss_type=freq_loss_type,
                high_freq_weight=high_freq_weight
            )
        
        if self.use_perceptual:
            self.perceptual_loss = PerceptualConsistencyLoss(
                loss_type=perceptual_loss_type
            )
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        计算总的一致性损失
        Args:
            pred: (B, C, H, W) 预测图像
            target: (B, C, H, W) 目标图像
            return_details: 是否返回各项损失的详细信息
        Returns:
            loss: 标量损失（或损失字典）
        """
        total_loss = 0.0
        loss_dict = {}
        
        if self.use_edge:
            edge_loss = self.edge_loss(pred, target)
            total_loss += self.edge_weight * edge_loss
            loss_dict['edge'] = edge_loss.item()
        
        if self.use_frequency:
            freq_loss = self.freq_loss(pred, target)
            total_loss += self.frequency_weight * freq_loss
            loss_dict['frequency'] = freq_loss.item()
        
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(pred, target)
            total_loss += self.perceptual_weight * perceptual_loss
            loss_dict['perceptual'] = perceptual_loss.item()
        
        if return_details:
            loss_dict['total_consistency'] = total_loss.item()
            return total_loss, loss_dict
        
        return total_loss


def create_consistency_loss_manager(args) -> Optional[ConsistencyLossManager]:
    """
    从训练参数创建一致性损失管理器
    Args:
        args: 训练参数
    Returns:
        manager: ConsistencyLossManager实例，或None（如果未启用）
    """
    if not getattr(args, 'use_consistency_loss', False):
        return None
    
    use_edge = getattr(args, 'consistency_use_edge', False)
    use_frequency = getattr(args, 'consistency_use_frequency', False)
    use_perceptual = getattr(args, 'consistency_use_perceptual', False)

    # If user enables consistency loss but doesn't specify components,
    # default to edge + frequency for a lightweight setup.
    if not any([use_edge, use_frequency, use_perceptual]):
        use_edge = True
        use_frequency = True

    manager = ConsistencyLossManager(
        use_edge=use_edge,
        use_frequency=use_frequency,
        use_perceptual=use_perceptual,
        edge_weight=getattr(args, 'consistency_edge_weight', 1.0),
        frequency_weight=getattr(args, 'consistency_frequency_weight', 1.0),
        perceptual_weight=getattr(args, 'consistency_perceptual_weight', 0.1),
        edge_loss_type=getattr(args, 'consistency_edge_loss_type', 'l1'),
        freq_loss_type=getattr(args, 'consistency_freq_loss_type', 'l1'),
        perceptual_loss_type=getattr(args, 'consistency_perceptual_loss_type', 'l1'),
        high_freq_weight=getattr(args, 'consistency_high_freq_weight', 2.0),
    )
    
    return manager

