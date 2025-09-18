"""训练配置类"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class OptimizerConfig:
    """优化器配置"""
    
    # 优化器类型
    name: str = "AdamW"  # Adam, AdamW, SGD
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Adam特有参数
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # SGD特有参数
    momentum: float = 0.9
    nesterov: bool = False


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    
    # 调度器类型
    name: str = "cosine"  # cosine, linear, exponential, step
    
    # 不同调度器的参数
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    
    # cosine调度器
    eta_min: float = 1e-6
    
    # step调度器
    step_size: int = 10000
    gamma: float = 0.5
    
    # 指数调度器
    decay_rate: float = 0.95


@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 训练基本参数
    num_epochs: int = 100
    max_steps: Optional[int] = None  # 如果设置，优先级高于num_epochs
    
    # 数据加载参数
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # 梯度累积
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0  # 梯度裁剪
    
    # 保存和验证
    save_every: int = 1000  # 步数
    eval_every: int = 500   # 步数
    log_every: int = 100    # 步数
    
    # 早停
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # 模型平均
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # 损失函数权重
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "diffusion_loss": 1.0,
        "prediction_loss": 0.1
    })
    
    # 优化器和调度器
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # 数据增强
    use_data_augmentation: bool = True
    augmentation_params: Dict[str, Any] = field(default_factory=lambda: {
        "noise_prob": 0.1,
        "mask_prob": 0.05,
        "reverse_prob": 0.02
    })
    
    # 正则化
    use_spectral_norm: bool = False
    dropout_rate: float = 0.1
    
    # 数值稳定性
    use_mixed_precision: bool = True
    loss_scale: str = "dynamic"  # dynamic, static, float
    
    # 分布式训练
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    def get_optimizer_params(self) -> Dict[str, Any]:
        """获取优化器参数"""
        params = self.optimizer.__dict__.copy()
        optimizer_name = params.pop('name')
        return optimizer_name, params
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """获取调度器参数"""
        params = self.scheduler.__dict__.copy()
        scheduler_name = params.pop('name')
        return scheduler_name, params
