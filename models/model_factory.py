"""模型工厂 - 统一的模型创建接口"""

from typing import Union, Dict, Any, Optional
import torch
import torch.nn as nn

from .transformer_predictor import (
    TransformerPredictor, 
    TransformerConfig,
    create_transformer_predictor
)
from .predictor_interface import UniversalPredictor
from ..config.transformer_config import TransformerPredictorConfig
from ..utils.logger import get_logger

try:
    from ..core.predictor_model import PromoterPredictor
    from ..config.model_config import PredictorModelConfig
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning("CNN预测器不可用，仅支持Transformer模型")

logger = get_logger(__name__)


class PredictorModelFactory:
    """预测器模型工厂"""
    
    @staticmethod
    def create_predictor(
        model_type: str = "transformer",
        config: Union[Dict[str, Any], TransformerPredictorConfig, Any] = None,
        return_universal: bool = True,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> Union[nn.Module, UniversalPredictor]:
        """
        创建预测器模型
        
        Args:
            model_type: 模型类型 ("transformer", "cnn", "hybrid")
            config: 模型配置
            return_universal: 是否返回统一接口包装
            pretrained_path: 预训练模型路径
            **kwargs: 额外参数
            
        Returns:
            预测器模型实例
        """
        
        if model_type == "transformer":
            model = PredictorModelFactory._create_transformer_predictor(config, pretrained_path, **kwargs)
        elif model_type == "cnn" and CNN_AVAILABLE:
            model = PredictorModelFactory._create_cnn_predictor(config, **kwargs)
        elif model_type == "hybrid":
            model = PredictorModelFactory._create_hybrid_predictor(config, pretrained_path, **kwargs)
        else:
            if not CNN_AVAILABLE and model_type == "cnn":
                logger.warning("CNN模型不可用，转为Transformer模型")
                model = PredictorModelFactory._create_transformer_predictor(config, pretrained_path, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        if return_universal:
            return UniversalPredictor(model, model_type)
        else:
            return model
    
    @staticmethod
    def _create_transformer_predictor(
        config: Union[Dict[str, Any], TransformerPredictorConfig] = None,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> TransformerPredictor:
        """创建Transformer预测器"""
        
        if config is None:
            config = TransformerPredictorConfig()
        elif isinstance(config, dict):
            config = TransformerPredictorConfig(**config)
        
        # 转换为TransformerConfig
        transformer_config = TransformerConfig(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_dim=config.intermediate_dim,
            dropout=config.dropout,
            max_position_embeddings=config.max_position_embeddings,
            position_embedding_type=config.position_embedding_type,
            layer_norm_eps=config.layer_norm_eps,
            feature_fusion_method=config.feature_fusion_method,
            feature_hidden_dim=config.feature_hidden_dim,
            num_classes=config.num_classes,
            output_activation=config.output_activation,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            hidden_dropout_prob=config.hidden_dropout_prob,
            use_kmer_features=config.use_kmer_features,
            use_biological_features=config.use_biological_features,
            kmer_sizes=config.kmer_sizes,
            use_rotary_embeddings=config.use_rotary_embeddings,
            rotary_theta=config.rotary_theta
        )
        
        model = create_transformer_predictor(transformer_config, pretrained_path)
        logger.info(f"Created Transformer predictor with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    @staticmethod
    def _create_cnn_predictor(
        config: Union[Dict[str, Any], Any] = None,
        **kwargs
    ) -> nn.Module:
        """创建CNN预测器（原有模型）"""
        
        if not CNN_AVAILABLE:
            raise ImportError("CNN predictor not available")
        
        if config is None:
            config = PredictorModelConfig()
        elif isinstance(config, dict):
            config = PredictorModelConfig(**config)
        
        model = PromoterPredictor(config)
        logger.info(f"Created CNN predictor with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    @staticmethod
    def _create_hybrid_predictor(
        config: Union[Dict[str, Any], TransformerPredictorConfig] = None,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """创建混合模型预测器"""
        
        # 这里可以实现Transformer + CNN的混合架构
        # 目前返回Transformer作为默认实现
        logger.warning("Hybrid model not implemented, falling back to Transformer")
        return PredictorModelFactory._create_transformer_predictor(config, pretrained_path, **kwargs)
    
    @staticmethod
    def create_optimized_config(performance_level: str = "high") -> TransformerPredictorConfig:
        """创建优化配置"""
        
        if performance_level == "high":
            return TransformerPredictorConfig(
                vocab_size=8,
                hidden_dim=768,
                num_layers=12,
                num_heads=12,
                intermediate_dim=3072,
                dropout=0.1,
                max_position_embeddings=2048,
                feature_fusion_method="attention",
                use_gradient_checkpointing=True,
                use_rotary_embeddings=True
            )
        elif performance_level == "medium":
            return TransformerPredictorConfig(
                vocab_size=8,
                hidden_dim=512,
                num_layers=8,
                num_heads=8,
                intermediate_dim=2048,
                dropout=0.1,
                max_position_embeddings=1024,
                feature_fusion_method="attention"
            )
        elif performance_level == "low":
            return TransformerPredictorConfig(
                vocab_size=8,
                hidden_dim=256,
                num_layers=4,
                num_heads=4,
                intermediate_dim=1024,
                dropout=0.1,
                max_position_embeddings=512,
                feature_fusion_method="concat",
                use_kmer_features=False,
                use_biological_features=False
            )
        else:
            raise ValueError(f"Unknown performance level: {performance_level}")


class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "model_type": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_footprint_mb": total_params * 4 / (1024 * 1024),  # 假设float32
        }
    
    @staticmethod
    def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
        """比较两个模型"""
        info1 = ModelUtils.get_model_info(model1)
        info2 = ModelUtils.get_model_info(model2)
        
        return {
            "model1": info1,
            "model2": info2,
            "parameter_ratio": info1["total_parameters"] / info2["total_parameters"],
            "memory_ratio": info1["memory_footprint_mb"] / info2["memory_footprint_mb"]
        }
    
    @staticmethod
    def optimize_model_for_inference(model: nn.Module) -> nn.Module:
        """优化模型用于推理"""
        model.eval()
        
        # 如果支持，编译模型
        try:
            if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
                model = torch.compile(model, mode="max-autotune")
                logger.info("Model compiled for optimized inference")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
        
        return model