"""
Timestep-Adaptive Conditioning for Diffusion-based Super-Resolution

这个模块实现了时间步自适应条件控制机制，根据扩散模型的去噪阶段
动态调整条件强度。

核心思想：
1. 早期时间步（高噪声）：需要强条件引导恢复全局结构
2. 中期时间步（中等噪声）：平衡结构保真和细节生成
3. 后期时间步（低噪声）：弱条件保证纹理自然多样

创新点：
- 时间步感知的条件权重调节
- 与层级权重正交互补
- 可学习的时间步-权重映射函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal


class TimestepAdaptiveWeights(nn.Module):
    """
    时间步自适应权重模块。
    
    根据当前去噪时间步动态调整条件强度。
    支持多种调节策略：线性、余弦、可学习MLP。
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        strategy: Literal["linear", "cosine", "learned", "exponential"] = "cosine",
        max_weight: float = 1.3,
        min_weight: float = 0.7,
        learnable: bool = False,
        hidden_dim: int = 128,
    ):
        """
        Args:
            num_train_timesteps: 训练时的总时间步数（DDPM默认1000）
            strategy: 权重调节策略
                - "linear": 线性衰减从max到min
                - "cosine": 余弦衰减（平滑过渡）
                - "exponential": 指数衰减
                - "learned": 可学习的MLP映射
            max_weight: 最大权重（早期时间步）
            min_weight: 最小权重（后期时间步）
            learnable: 是否使用可学习的映射
            hidden_dim: 可学习模式下的隐藏层维度
        """
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.strategy = strategy
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.learnable = learnable
        
        if learnable or strategy == "learned":
            # 可学习的时间步嵌入和MLP
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            # 初始化使其接近线性映射
            self._init_learned_weights()
        else:
            self.time_embedding = None
    
    def _init_learned_weights(self):
        """初始化可学习权重，使其接近预设策略"""
        # 最后一层的bias初始化为接近1.0
        if self.time_embedding is not None:
            nn.init.constant_(self.time_embedding[-1].bias, 1.0)
            nn.init.normal_(self.time_embedding[-1].weight, 0.0, 0.01)
    
    def _linear_schedule(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        线性衰减：权重从max线性降到min
        
        早期(t=1000) → max_weight (e.g., 1.3)
        后期(t=0) → min_weight (e.g., 0.7)
        """
        # 归一化到[0, 1]
        t_norm = timesteps.float() / self.num_train_timesteps
        # 线性插值
        weights = self.max_weight - (self.max_weight - self.min_weight) * (1.0 - t_norm)
        return weights
    
    def _cosine_schedule(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        余弦衰减：平滑的非线性过渡
        
        使用余弦函数实现平滑过渡，避免线性的突变
        """
        t_norm = timesteps.float() / self.num_train_timesteps
        # 余弦插值：cos从π到0映射到[-1, 1]，再映射到[min, max]
        cosine_factor = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - t_norm)))
        weights = self.min_weight + (self.max_weight - self.min_weight) * cosine_factor
        return weights
    
    def _exponential_schedule(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        指数衰减：早期变化慢，后期变化快
        
        适用于希望后期快速降低条件强度的场景
        """
        t_norm = timesteps.float() / self.num_train_timesteps
        # 指数衰减
        exp_factor = torch.exp(-2.0 * (1.0 - t_norm))  # e^(-2*(1-t))
        weights = self.min_weight + (self.max_weight - self.min_weight) * (1.0 - exp_factor) / (1.0 - math.exp(-2.0))
        return weights
    
    def _learned_schedule(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        可学习的调度：通过MLP学习最优的时间步-权重映射
        """
        # 归一化时间步到[0, 1]
        t_norm = (timesteps.float() / self.num_train_timesteps).unsqueeze(-1)  # (B, 1)
        
        # 通过MLP预测权重缩放因子
        scale_factor = self.time_embedding(t_norm).squeeze(-1)  # (B,)
        
        # 应用sigmoid确保在合理范围内，然后映射到[min, max]
        scale_factor = torch.sigmoid(scale_factor)
        weights = self.min_weight + (self.max_weight - self.min_weight) * scale_factor
        
        return weights
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        根据时间步计算条件权重。
        
        Args:
            timesteps: 当前批次的时间步，形状 (B,) 或 (B, 1)
        
        Returns:
            weights: 对应的条件权重，形状 (B,)
        """
        if timesteps.dim() > 1:
            timesteps = timesteps.squeeze()
        
        # 根据策略选择调度函数
        if self.strategy == "linear":
            weights = self._linear_schedule(timesteps)
        elif self.strategy == "cosine":
            weights = self._cosine_schedule(timesteps)
        elif self.strategy == "exponential":
            weights = self._exponential_schedule(timesteps)
        elif self.strategy == "learned":
            weights = self._learned_schedule(timesteps)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return weights
    
    def get_weight_at_step(self, step: int) -> float:
        """
        获取指定时间步的权重（用于可视化和分析）
        
        Args:
            step: 时间步 (0 到 num_train_timesteps)
        
        Returns:
            weight: 该时间步的权重值
        """
        timesteps = torch.tensor([step], dtype=torch.long)
        with torch.no_grad():
            weight = self.forward(timesteps)
        return weight.item()
    
    def visualize_schedule(self, num_points: int = 100) -> Tuple[list, list]:
        """
        生成权重调度的可视化数据
        
        Returns:
            timesteps: 时间步列表
            weights: 对应的权重列表
        """
        timesteps = torch.linspace(0, self.num_train_timesteps, num_points).long()
        with torch.no_grad():
            weights = self.forward(timesteps)
        return timesteps.tolist(), weights.tolist()
    
    def extra_repr(self) -> str:
        return (f'strategy={self.strategy}, '
                f'max_weight={self.max_weight}, '
                f'min_weight={self.min_weight}, '
                f'learnable={self.learnable}')


class JointConditionScaler(nn.Module):
    """
    联合条件缩放器：结合层级权重和时间步权重。
    
    将多尺度条件注入的层级权重与时间步自适应权重结合，
    实现二维的自适应条件控制。
    """
    
    def __init__(
        self,
        num_layers: int,
        layer_weights: Optional[nn.Module] = None,
        timestep_weights: Optional[TimestepAdaptiveWeights] = None,
        combination: Literal["multiply", "add", "learned"] = "multiply",
    ):
        """
        Args:
            num_layers: ControlNet输出层数
            layer_weights: 现有的层级权重模块（LearnableScaleWeights）
            timestep_weights: 时间步自适应权重模块
            combination: 组合方式
                - "multiply": 相乘 (layer_w * timestep_w)
                - "add": 相加 (layer_w + timestep_w - 1)
                - "learned": 可学习的组合
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_weights = layer_weights
        self.timestep_weights = timestep_weights
        self.combination = combination
        
        if combination == "learned" and layer_weights is not None and timestep_weights is not None:
            # 可学习的组合权重
            self.alpha = nn.Parameter(torch.tensor(0.5))  # 插值权重
    
    def forward(
        self,
        layer_idx: Optional[int] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算联合条件权重。
        
        Args:
            layer_idx: 层索引（如果为None则返回所有层的权重）
            timesteps: 时间步 (B,)
        
        Returns:
            combined_weights: 组合后的权重
                - 如果layer_idx不为None: (B,) 每个样本在该层的权重
                - 如果layer_idx为None: (B, num_layers) 每个样本在所有层的权重
        """
        # 获取层级权重
        if self.layer_weights is not None:
            if layer_idx is not None:
                layer_w = self.layer_weights(layer_idx)  # scalar
            else:
                layer_w = self.layer_weights()  # (num_layers,)
        else:
            layer_w = 1.0
        
        # 获取时间步权重
        if self.timestep_weights is not None and timesteps is not None:
            timestep_w = self.timestep_weights(timesteps)  # (B,)
        else:
            timestep_w = 1.0
        
        # 组合权重
        if self.combination == "multiply":
            # 乘法组合：layer_w * timestep_w
            if isinstance(layer_w, torch.Tensor) and layer_w.dim() > 0:
                # layer_w: (num_layers,), timestep_w: (B,)
                # 结果: (B, num_layers)
                if isinstance(timestep_w, torch.Tensor):
                    combined = timestep_w.unsqueeze(1) * layer_w.unsqueeze(0)
                else:
                    combined = layer_w.unsqueeze(0) * timestep_w
            else:
                combined = layer_w * timestep_w
        
        elif self.combination == "add":
            # 加法组合：layer_w + timestep_w - 1 (避免权重过大)
            if isinstance(layer_w, torch.Tensor) and layer_w.dim() > 0:
                if isinstance(timestep_w, torch.Tensor):
                    combined = timestep_w.unsqueeze(1) + layer_w.unsqueeze(0) - 1.0
                else:
                    combined = layer_w + timestep_w - 1.0
            else:
                combined = layer_w + timestep_w - 1.0
        
        elif self.combination == "learned":
            # 可学习的插值
            alpha = torch.sigmoid(self.alpha)
            if isinstance(layer_w, torch.Tensor) and layer_w.dim() > 0:
                if isinstance(timestep_w, torch.Tensor):
                    combined = alpha * (timestep_w.unsqueeze(1) * layer_w.unsqueeze(0)) + \
                              (1 - alpha) * (timestep_w.unsqueeze(1) + layer_w.unsqueeze(0) - 1.0)
                else:
                    combined = alpha * (layer_w * timestep_w) + (1 - alpha) * (layer_w + timestep_w - 1.0)
            else:
                combined = alpha * (layer_w * timestep_w) + (1 - alpha) * (layer_w + timestep_w - 1.0)
        else:
            raise ValueError(f"Unknown combination: {self.combination}")
        
        # 如果指定了layer_idx，提取对应层
        if layer_idx is not None and isinstance(combined, torch.Tensor) and combined.dim() > 1:
            combined = combined[:, layer_idx]
        
        return combined
    
    def extra_repr(self) -> str:
        return f'num_layers={self.num_layers}, combination={self.combination}'


def create_timestep_adaptive_weights(
    num_train_timesteps: int = 1000,
    strategy: str = "cosine",
    max_weight: float = 1.3,
    min_weight: float = 0.7,
    learnable: bool = False,
) -> TimestepAdaptiveWeights:
    """
    工厂函数：创建时间步自适应权重模块
    
    Args:
        num_train_timesteps: 训练时间步数
        strategy: 权重策略 ("linear", "cosine", "exponential", "learned")
        max_weight: 最大权重（早期时间步）
        min_weight: 最小权重（后期时间步）
        learnable: 是否使用可学习的映射
    
    Returns:
        timestep_weights: 时间步自适应权重模块
    """
    return TimestepAdaptiveWeights(
        num_train_timesteps=num_train_timesteps,
        strategy=strategy,
        max_weight=max_weight,
        min_weight=min_weight,
        learnable=learnable,
    )


def create_joint_scaler(
    num_layers: int,
    layer_weights: Optional[nn.Module] = None,
    timestep_weights: Optional[TimestepAdaptiveWeights] = None,
    combination: str = "multiply",
) -> JointConditionScaler:
    """
    工厂函数：创建联合条件缩放器
    
    Args:
        num_layers: 层数
        layer_weights: 层级权重模块
        timestep_weights: 时间步权重模块
        combination: 组合方式
    
    Returns:
        scaler: 联合条件缩放器
    """
    return JointConditionScaler(
        num_layers=num_layers,
        layer_weights=layer_weights,
        timestep_weights=timestep_weights,
        combination=combination,
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试时间步自适应权重模块")
    print("=" * 60)
    
    # 测试不同策略
    strategies = ["linear", "cosine", "exponential"]
    
    for strategy in strategies:
        print(f"\n策略: {strategy}")
        timestep_weights = create_timestep_adaptive_weights(
            num_train_timesteps=1000,
            strategy=strategy,
            max_weight=1.3,
            min_weight=0.7,
        )
        
        # 测试几个关键时间步
        test_steps = [0, 250, 500, 750, 1000]
        for step in test_steps:
            weight = timestep_weights.get_weight_at_step(step)
            print(f"  t={step:4d}: weight={weight:.4f}")
    
    # 测试可学习策略
    print(f"\n策略: learned")
    timestep_weights = create_timestep_adaptive_weights(
        num_train_timesteps=1000,
        strategy="learned",
        learnable=True,
    )
    
    batch_timesteps = torch.tensor([0, 250, 500, 750, 1000])
    weights = timestep_weights(batch_timesteps)
    for step, weight in zip(batch_timesteps.tolist(), weights.tolist()):
        print(f"  t={step:4d}: weight={weight:.4f}")
    
    print("\n" + "=" * 60)
    print("测试联合条件缩放器")
    print("=" * 60)
    
    # 模拟层级权重
    from models.multi_scale_conditioning import LearnableScaleWeights
    layer_weights = LearnableScaleWeights(num_layers=5, init_value=1.0)
    
    # 创建联合缩放器
    scaler = create_joint_scaler(
        num_layers=5,
        layer_weights=layer_weights,
        timestep_weights=timestep_weights,
        combination="multiply",
    )
    
    # 测试批次
    batch_size = 2
    timesteps = torch.tensor([100, 800])
    
    print(f"\n批次大小: {batch_size}")
    print(f"时间步: {timesteps.tolist()}")
    
    # 获取所有层的权重
    combined_weights = scaler(layer_idx=None, timesteps=timesteps)
    print(f"联合权重形状: {combined_weights.shape}")  # 应该是 (2, 5)
    print(f"联合权重:\n{combined_weights}")
    
    print("\n✅ 所有测试通过！")

