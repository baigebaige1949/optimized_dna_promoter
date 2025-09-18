"""
Dirichlet扩散模型配置
基于DDSM论文的参数配置
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class DirichletDiffusionConfig:
    """Dirichlet扩散模型配置类"""
    
    # 模型架构参数
    sequence_length: int = 1000
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    
    # Jacobi过程参数
    alpha: float = 2.0  # Jacobi过程参数α
    beta: float = 2.0   # Jacobi过程参数β
    
    # 时间膨胀参数
    dilation_factor: float = 2.0
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    
    # 采样参数
    num_sampling_steps: int = 100
    temperature: float = 1.0
    
    # 损失权重
    score_loss_weight: float = 1.0
    importance_sampling: bool = True
    
    # 设备和优化
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # 评估参数
    eval_interval: int = 10
    save_interval: int = 20
    num_eval_samples: int = 100
    
    def __post_init__(self):
        """配置验证"""
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Jacobi参数α和β必须为正数")
        if self.dilation_factor <= 0:
            raise ValueError("时间膨胀因子必须为正数")
        if self.temperature <= 0:
            raise ValueError("采样温度必须为正数")

# 预定义配置
DEFAULT_CONFIG = DirichletDiffusionConfig()

# 快速训练配置（用于调试）
FAST_CONFIG = DirichletDiffusionConfig(
    sequence_length=100,
    hidden_dim=128,
    num_layers=4,
    batch_size=8,
    num_epochs=10,
    num_sampling_steps=50
)

# 高质量配置（用于生产）
HIGH_QUALITY_CONFIG = DirichletDiffusionConfig(
    sequence_length=1000,
    hidden_dim=512,
    num_layers=12,
    num_heads=16,
    batch_size=8,
    num_epochs=200,
    num_sampling_steps=200,
    temperature=0.8
)