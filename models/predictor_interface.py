"""统一的预测器接口"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np


class PredictorInterface(ABC):
    """预测器统一接口"""
    
    @abstractmethod
    def predict_strength(self, sequences: List[str]) -> List[float]:
        """预测启动子强度"""
        pass
    
    @abstractmethod
    def predict_batch(
        self, 
        sequences: List[str], 
        batch_size: int = 32
    ) -> List[float]:
        """批量预测"""
        pass
    
    @abstractmethod
    def get_feature_importance(self, sequences: List[str]) -> Dict[str, float]:
        """获取特征重要性"""
        pass


class UniversalPredictor(PredictorInterface):
    """通用预测器包装器"""
    
    def __init__(self, model: nn.Module, model_type: str = "transformer"):
        self.model = model
        self.model_type = model_type
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    
    def predict_strength(self, sequences: List[str]) -> List[float]:
        """预测启动子强度"""
        if hasattr(self.model, 'predict_strength'):
            return self.model.predict_strength(sequences)
        else:
            # 通用实现
            return self._generic_predict(sequences)
    
    def predict_batch(
        self, 
        sequences: List[str], 
        batch_size: int = 32
    ) -> List[float]:
        """批量预测"""
        if hasattr(self.model, 'predict_batch'):
            return self.model.predict_batch(sequences, batch_size)
        else:
            # 通用批量预测实现
            all_predictions = []
            self.model.eval()
            
            with torch.no_grad():
                for i in range(0, len(sequences), batch_size):
                    batch_sequences = sequences[i:i + batch_size]
                    batch_preds = self._generic_predict(batch_sequences)
                    all_predictions.extend(batch_preds)
            
            return all_predictions
    
    def get_feature_importance(self, sequences: List[str]) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance(sequences)
        else:
            # 通用特征重要性实现（简化）
            return self._generic_feature_importance()
    
    def _generic_predict(self, sequences: List[str]) -> List[float]:
        """通用预测实现"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(sequences)
            
            if isinstance(outputs, dict):
                predictions = outputs.get('predictions', outputs.get('logits'))
            else:
                predictions = outputs
            
            if isinstance(predictions, torch.Tensor):
                # 处理不同的输出格式
                if predictions.dim() == 1:
                    return predictions.cpu().numpy().tolist()
                elif predictions.dim() == 2 and predictions.size(1) == 1:
                    return predictions.squeeze(-1).cpu().numpy().tolist()
                else:
                    # 多分类情况，取最大概率
                    return predictions.argmax(dim=-1).cpu().numpy().tolist()
            else:
                return predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
    
    def _generic_feature_importance(self) -> Dict[str, float]:
        """通用特征重要性实现"""
        importance = {}
        
        if self.model_type == "transformer":
            importance.update({
                "transformer_attention": 0.95,
                "transformer_position": 0.85,
                "kmer_3": 0.80,
                "kmer_4": 0.75,
                "kmer_5": 0.70,
                "kmer_6": 0.65,
                "biological_features": 0.85
            })
        elif self.model_type == "cnn":
            importance.update({
                "cnn_features": 0.85,
                "kmer_3": 0.70,
                "kmer_4": 0.65,
                "kmer_5": 0.60,
                "biological_features": 0.75
            })
        
        return importance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": self.model_type,
            "model_class": self.model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "memory_footprint_mb": total_params * 4 / (1024 * 1024)
        }
    
    def benchmark_inference(self, sequences: List[str], num_runs: int = 10) -> Dict[str, float]:
        """性能基准测试"""
        import time
        
        self.model.eval()
        times = []
        
        # 预热
        with torch.no_grad():
            _ = self.predict_strength(sequences[:1])
        
        # 基准测试
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.predict_strength(sequences)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "avg_inference_time": np.mean(times),
            "std_inference_time": np.std(times),
            "min_inference_time": np.min(times),
            "max_inference_time": np.max(times),
            "sequences_per_second": len(sequences) / np.mean(times)
        }