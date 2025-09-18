"""核心模型模块"""

from .diffusion_model import DiffusionModel, NoiseScheduler
from .conditional_diffusion_model import (
    ConditionalDiffusionModel,
    ConditionConfig,
    ConditionEmbedding,
    ConditionalUNet1D,
    CrossAttention1D,
    ConditionValidator,
    create_conditional_diffusion_model
)
# from .predictor_model import PredictorModel
from .feature_extractor import FeatureExtractor

__all__ = [
    # 基础扩散模型
    'DiffusionModel',
    'NoiseScheduler',
    
    # 条件扩散模型
    'ConditionalDiffusionModel',
    'ConditionConfig', 
    'ConditionEmbedding',
    'ConditionalUNet1D',
    'CrossAttention1D',
    'ConditionValidator',
    'create_conditional_diffusion_model',
    
    # 其他模型
    'PromoterPredictor', 'PredictorModel',
    'FeatureExtractor'
]
# ---- fixed: export real predictor and a compat alias ----
from .predictor_model import PromoterPredictor
PredictorModel = PromoterPredictor  # backward-compat alias
