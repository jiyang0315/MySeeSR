"""
Multi-Scale Conditional Injection for ControlNet-based Super-Resolution

这个模块实现了多尺度条件注入机制，允许在UNet的不同层级注入
不同尺度的条件信息，并使用可学习的权重来平衡各层的条件强度。

创新点：
1. 自适应的层级条件强度：每层可以学习最优的条件权重
2. 多尺度特征金字塔：不同层接收相应分辨率的条件
3. 渐进式约束：从粗到细的层级式超分辨率引导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any


class LearnableScaleWeights(nn.Module):
    """
    可学习的多尺度条件权重。
    
    为ControlNet的每个输出层学习一个权重，用于动态调整该层条件的影响强度。
    初始化为接近1.0，训练过程中可以自适应调整。
    """
    
    def __init__(
        self,
        num_layers: int,
        init_value: float = 1.0,
        learnable: bool = True,
    ):
        """
        Args:
            num_layers: ControlNet输出层的数量（通常是down blocks + mid block）
            init_value: 初始权重值
            learnable: 是否允许训练时更新权重
        """
        super().__init__()
        self.num_layers = num_layers
        self.learnable = learnable
        
        # 初始化可学习权重参数
        weights = torch.ones(num_layers) * init_value
        
        if learnable:
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer('weights', weights)
    
    def forward(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        获取指定层的权重，或返回所有权重。
        
        Args:
            layer_idx: 层索引，如果为None则返回所有权重
        
        Returns:
            weight(s): 标量或向量
        """
        if layer_idx is not None:
            return self.weights[layer_idx]
        return self.weights
    
    def get_all_weights(self) -> torch.Tensor:
        """返回所有层的权重"""
        return self.weights
    
    def extra_repr(self) -> str:
        return f'num_layers={self.num_layers}, learnable={self.learnable}'


class DynamicScalePredictor(nn.Module):
    """
    动态尺度预测器。
    根据输入条件特征预测每个层级的条件强度（每个样本独立）。
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        hidden_dim: int = 128,
        min_scale: float = 0.5,
        max_scale: float = 1.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = self._build_mlp(self.num_layers)

    def _build_mlp(self, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, out_dim),
        )

    def resize_num_layers(self, new_num_layers: int):
        if new_num_layers == self.num_layers:
            return
        self.num_layers = new_num_layers
        self.mlp = self._build_mlp(self.num_layers).to(next(self.parameters()).device)

    def forward(self, cond_features: torch.Tensor) -> torch.Tensor:
        # cond_features: (B, C, H, W)
        x = self.pool(cond_features).flatten(1)
        x = x.to(dtype=next(self.mlp.parameters()).dtype)
        logits = self.mlp(x)
        weights = torch.sigmoid(logits)
        weights = self.min_scale + (self.max_scale - self.min_scale) * weights
        return weights


class MultiScaleConditionInjector(nn.Module):
    """
    多尺度条件注入器。
    
    这个模块负责：
    1. 管理多尺度条件特征
    2. 在不同UNet层级应用相应尺度的条件
    3. 使用可学习权重调节条件强度
    """
    
    def __init__(
        self,
        num_down_blocks: int = 4,
        has_mid_block: bool = True,
        learnable_scales: bool = True,
        init_scale: float = 1.0,
        progressive_scale: bool = True,
        use_dynamic_scales: bool = False,
        dynamic_in_channels: int = 3,
        dynamic_hidden_dim: int = 128,
        dynamic_min_scale: float = 0.5,
        dynamic_max_scale: float = 1.5,
    ):
        """
        Args:
            num_down_blocks: UNet下采样块的数量（初始估计，会自动调整）
            has_mid_block: 是否有中间块
            learnable_scales: 是否使用可学习的尺度权重
            init_scale: 初始尺度值
            progressive_scale: 是否使用渐进式尺度（深层条件更弱）
        """
        super().__init__()
        self.num_down_blocks = num_down_blocks
        self.has_mid_block = has_mid_block
        self.progressive_scale = progressive_scale
        self.learnable_scales = learnable_scales
        self.init_scale = init_scale
        self.use_dynamic_scales = use_dynamic_scales
        self.dynamic_in_channels = dynamic_in_channels
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.dynamic_min_scale = dynamic_min_scale
        self.dynamic_max_scale = dynamic_max_scale
        self._initialized = False
        self._last_dynamic_weights_mean = None
        
        # 计算总层数（初始估计）
        total_layers = num_down_blocks + (1 if has_mid_block else 0)
        
        # 初始化可学习权重
        if progressive_scale and not learnable_scales:
            # 使用渐进式固定权重：浅层强，深层弱
            # 例如：[1.2, 1.1, 1.0, 0.9, 0.8]
            init_weights = torch.linspace(init_scale * 1.2, init_scale * 0.8, total_layers)
            self.scale_weights = LearnableScaleWeights(
                num_layers=total_layers,
                init_value=init_weights[0],  # dummy
                learnable=False
            )
            self.scale_weights.weights.copy_(init_weights)
        else:
            self.scale_weights = LearnableScaleWeights(
                num_layers=total_layers,
                init_value=init_scale,
                learnable=learnable_scales
            )
        
        self.total_layers = total_layers
        if self.use_dynamic_scales:
            self.dynamic_predictor = DynamicScalePredictor(
                in_channels=self.dynamic_in_channels,
                num_layers=total_layers,
                hidden_dim=self.dynamic_hidden_dim,
                min_scale=self.dynamic_min_scale,
                max_scale=self.dynamic_max_scale,
            )
        else:
            self.dynamic_predictor = None
    
    def _resize_weights(self, new_size: int):
        """
        动态调整权重数量以匹配实际的层数。
        """
        old_weights = self.scale_weights.get_all_weights()
        old_size = len(old_weights)
        
        if old_size == new_size:
            return
        
        # 创建新的权重
        if self.progressive_scale and not self.learnable_scales:
            # 重新生成渐进式权重
            new_weights = torch.linspace(
                self.init_scale * 1.2, 
                self.init_scale * 0.8, 
                new_size
            )
        else:
            # 插值旧权重或使用初始值
            if old_size > 1:
                # 插值
                indices = torch.linspace(0, old_size - 1, new_size)
                new_weights = torch.zeros(new_size)
                for i, idx in enumerate(indices):
                    idx_low = int(idx)
                    idx_high = min(idx_low + 1, old_size - 1)
                    weight_low = old_weights[idx_low]
                    weight_high = old_weights[idx_high]
                    alpha = idx - idx_low
                    new_weights[i] = weight_low * (1 - alpha) + weight_high * alpha
            else:
                # 使用初始值
                new_weights = torch.ones(new_size) * self.init_scale
        
        # 重新创建权重模块
        self.scale_weights = LearnableScaleWeights(
            num_layers=new_size,
            init_value=self.init_scale,
            learnable=self.learnable_scales
        )
        self.scale_weights.weights.data.copy_(new_weights.to(self.scale_weights.weights.device))
        self.total_layers = new_size
        self._initialized = True
        if self.use_dynamic_scales and self.dynamic_predictor is not None:
            self.dynamic_predictor.resize_num_layers(new_size)
    
    def apply_conditioning(
        self,
        controlnet_outputs: List[torch.Tensor],
        mid_block_output: Optional[torch.Tensor] = None,
        cond_features: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        对ControlNet的输出应用可学习的尺度权重。
        
        Args:
            controlnet_outputs: ControlNet下采样块的输出列表
            mid_block_output: ControlNet中间块的输出（可选）
        
        Returns:
            scaled_outputs: 经过尺度调整的输出
            scaled_mid: 经过尺度调整的中间块输出
        """
        # 确定实际需要的权重数量
        num_outputs = len(controlnet_outputs)
        if mid_block_output is not None:
            expected_weights = num_outputs + 1
        else:
            expected_weights = num_outputs
        
        # 第一次调用时，自动调整权重大小
        if not self._initialized:
            current_size = len(self.scale_weights.get_all_weights())
            if current_size != expected_weights:
                import warnings
                msg = (
                    f"\n{'='*70}\n"
                    f"⚠️  Auto-resizing multi-scale weights: {current_size} → {expected_weights}\n"
                    f"{'='*70}\n"
                )
                if self.learnable_scales:
                    msg += (
                        f"NOTE: Since weights are learnable, new weights won't be in optimizer.\n"
                        f"RECOMMENDATION: Stop training (Ctrl+C) and restart for best results.\n"
                        f"The restart will use the correct size (13 layers) from the start.\n"
                        f"{'='*70}\n"
                    )
                warnings.warn(msg)
                self._resize_weights(expected_weights)
        
        if self.use_dynamic_scales:
            if cond_features is None:
                raise ValueError("`cond_features` is required when `use_dynamic_scales=True`.")
            if cond_features.dim() != 4:
                raise ValueError(
                    f"`cond_features` must be 4D tensor [B, C, H, W], got shape: {tuple(cond_features.shape)}"
                )

            dynamic_weights = self.dynamic_predictor(cond_features)
            if dynamic_weights.shape[1] != expected_weights:
                self._resize_weights(expected_weights)
                dynamic_weights = self.dynamic_predictor(cond_features)

            self._last_dynamic_weights_mean = dynamic_weights.detach().mean(dim=0)
            scaled_outputs = []
            for i, output in enumerate(controlnet_outputs):
                w = dynamic_weights[:, i].view(-1, 1, 1, 1).to(dtype=output.dtype, device=output.device)
                scaled_outputs.append(output * w)

            scaled_mid = None
            if mid_block_output is not None and self.has_mid_block:
                mid_weight = dynamic_weights[:, num_outputs].view(-1, 1, 1, 1).to(
                    dtype=mid_block_output.dtype, device=mid_block_output.device
                )
                scaled_mid = mid_block_output * mid_weight

            return scaled_outputs, scaled_mid

        weights = self.scale_weights.get_all_weights()
        
        # 再次检查（理论上不会触发，但保险起见）
        if len(weights) != expected_weights:
            import warnings
            warnings.warn(f"Weight size mismatch after resize. Using uniform scaling.")
            avg_weight = weights.mean()
            scaled_outputs = [output * avg_weight for output in controlnet_outputs]
            scaled_mid = mid_block_output * avg_weight if mid_block_output is not None else None
            return scaled_outputs, scaled_mid
        
        # 对每个下采样块输出应用权重
        scaled_outputs = []
        for i, output in enumerate(controlnet_outputs):
            weight = weights[i]
            scaled_output = output * weight
            scaled_outputs.append(scaled_output)
        
        # 对中间块应用权重
        scaled_mid = None
        if mid_block_output is not None and self.has_mid_block:
            mid_weight = weights[num_outputs]
            scaled_mid = mid_block_output * mid_weight
        
        return scaled_outputs, scaled_mid
    
    def get_scale_info(self) -> Dict[str, any]:
        """
        获取当前尺度权重信息，用于监控和可视化。
        
        Returns:
            info: 包含尺度权重统计的字典
        """
        if self.use_dynamic_scales and self._last_dynamic_weights_mean is not None:
            weights = self._last_dynamic_weights_mean.detach().cpu()
        else:
            weights = self.scale_weights.get_all_weights().detach().cpu()
        return {
            'weights': weights.tolist(),
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'min': weights.min().item(),
            'max': weights.max().item(),
            'is_dynamic': self.use_dynamic_scales,
        }
    
    def extra_repr(self) -> str:
        return (f'num_down_blocks={self.num_down_blocks}, '
                f'has_mid_block={self.has_mid_block}, '
                f'progressive_scale={self.progressive_scale}')


class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合模块（可选）。
    
    如果需要更复杂的融合策略，可以使用这个模块来融合不同尺度的特征。
    """
    
    def __init__(
        self,
        in_channels: int,
        num_scales: int = 3,
        fusion_type: str = "concat",  # "concat", "add", "attention"
    ):
        """
        Args:
            in_channels: 输入通道数
            num_scales: 尺度数量
            fusion_type: 融合方式
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            # 拼接后用1x1卷积降维
            self.fusion = nn.Conv2d(
                in_channels * num_scales,
                in_channels,
                kernel_size=1,
                bias=False
            )
        elif fusion_type == "attention":
            # 使用注意力机制融合
            self.attention = nn.Sequential(
                nn.Conv2d(in_channels * num_scales, num_scales, 1),
                nn.Softmax(dim=1)
            )
        elif fusion_type == "add":
            # 简单相加，不需要额外参数
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        融合多尺度特征。
        
        Args:
            features: 多尺度特征列表，每个元素shape为(B, C, H, W)
        
        Returns:
            fused: 融合后的特征 (B, C, H, W)
        """
        if self.fusion_type == "concat":
            # 拼接并降维
            concat = torch.cat(features, dim=1)
            fused = self.fusion(concat)
        elif self.fusion_type == "attention":
            # 注意力加权融合
            concat = torch.cat(features, dim=1)
            attn_weights = self.attention(concat)  # (B, num_scales, H, W)
            
            fused = 0
            for i, feat in enumerate(features):
                weight = attn_weights[:, i:i+1, :, :]  # (B, 1, H, W)
                fused = fused + feat * weight
        elif self.fusion_type == "add":
            # 简单相加
            fused = torch.stack(features, dim=0).mean(dim=0)
        
        return fused


def create_multi_scale_injector(
    num_down_blocks: int = 4,
    learnable_scales: bool = True,
    progressive_scale: bool = True,
    init_scale: float = 1.0,
    use_dynamic_scales: bool = False,
    dynamic_in_channels: int = 3,
    dynamic_hidden_dim: int = 128,
    dynamic_min_scale: float = 0.5,
    dynamic_max_scale: float = 1.5,
) -> MultiScaleConditionInjector:
    """
    工厂函数：创建多尺度条件注入器。
    
    Args:
        num_down_blocks: UNet下采样块数量
        learnable_scales: 是否使用可学习权重
        progressive_scale: 是否使用渐进式尺度
        init_scale: 初始尺度值
    
    Returns:
        injector: MultiScaleConditionInjector实例
    
    Example:
        >>> injector = create_multi_scale_injector(num_down_blocks=4)
        >>> # 在训练中使用
        >>> down_samples, mid_sample = controlnet(...)
        >>> scaled_down, scaled_mid = injector.apply_conditioning(down_samples, mid_sample)
    """
    return MultiScaleConditionInjector(
        num_down_blocks=num_down_blocks,
        has_mid_block=True,
        learnable_scales=learnable_scales,
        init_scale=init_scale,
        progressive_scale=progressive_scale,
        use_dynamic_scales=use_dynamic_scales,
        dynamic_in_channels=dynamic_in_channels,
        dynamic_hidden_dim=dynamic_hidden_dim,
        dynamic_min_scale=dynamic_min_scale,
        dynamic_max_scale=dynamic_max_scale,
    )

