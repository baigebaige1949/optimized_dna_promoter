"""模型配置类"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DiffusionModelConfig:
    """扩散模型配置"""
    
    # 模型架构参数
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # 扩散过程参数
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # linear, cosine
    
    # 噪声调度参数
    variance_type: str = "fixed_small"  # fixed_small, fixed_large, learned
    
    # 条件生成参数
    use_condition: bool = True
    condition_dim: int = 64
    
    # U-Net参数
    unet_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 256])
    attention_resolutions: List[int] = field(default_factory=lambda: [32, 16, 8])


@dataclass
class PredictorModelConfig:
    """预测模型配置"""
    
    # 特征提取器参数
    kmer_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    use_biological_features: bool = True
    
    # 预训练模型参数
    pretrained_model_name: str = "zhihan1996/DNA_bert_3"
    use_pretrained: bool = True
    freeze_pretrained: bool = False
    
    # 分类/回归头参数
    classifier_hidden_dim: int = 512
    classifier_dropout: float = 0.2
    num_classes: int = 1  # 回归任务
    
    # 特征融合参数
    feature_fusion_method: str = "concat"  # concat, attention, gated


@dataclass
class ConditionalDiffusionModelConfig(DiffusionModelConfig):
    """条件扩散模型配置"""
    
    # 条件相关参数
    condition_embedding_dim: int = 256
    use_cross_attention: bool = True
    cross_attention_heads: int = 8
    
    # 无分类器引导参数
    use_classifier_free_guidance: bool = True
    cfg_drop_prob: float = 0.1
    default_guidance_scale: float = 7.5
    
    # 支持的条件类型
    supported_conditions: List[str] = field(default_factory=lambda: [
        'cell_type', 'temperature', 'ph', 'oxygen', 
        'growth_phase', 'nutrient_level', 'stress_level'
    ])
    
    # 条件验证参数
    validate_conditions: bool = True
    use_default_conditions: bool = True


@dataclass
class ModelConfig:
    """模型总配置"""
    
    # 子配置
    diffusion: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)
    conditional_diffusion: ConditionalDiffusionModelConfig = field(default_factory=ConditionalDiffusionModelConfig)
    predictor: PredictorModelConfig = field(default_factory=PredictorModelConfig)
    
    # 模型保存和加载
    save_checkpoint_every: int = 1000
    max_checkpoints_to_keep: int = 5
    
    # 模型优化参数
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0编译
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取模型参数字典"""
        return {
            "diffusion": self.diffusion.__dict__,
            "conditional_diffusion": self.conditional_diffusion.__dict__,
            "predictor": self.predictor.__dict__
        }
