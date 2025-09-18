"""Transformer模型专用配置"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TransformerPredictorConfig:
    """Transformer预测器配置"""
    
    # 模型架构
    vocab_size: int = 8
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_dim: int = 3072
    dropout: float = 0.1
    
    # 序列相关
    max_position_embeddings: int = 2048
    position_embedding_type: str = "absolute"
    layer_norm_eps: float = 1e-12
    
    # 特征融合
    feature_fusion_method: str = "attention"
    feature_hidden_dim: int = 256
    
    # 任务相关
    num_classes: int = 1
    output_activation: str = "sigmoid"
    
    # 性能优化
    use_gradient_checkpointing: bool = False
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    
    # 多维特征
    use_kmer_features: bool = True
    use_biological_features: bool = True
    kmer_sizes: List[int] = field(default_factory=lambda: [3, 4, 5, 6])
    
    # 位置编码
    use_rotary_embeddings: bool = True
    rotary_theta: float = 10000.0