# -*- coding: utf-8 -*-
"""多模态特征融合优化模块

实现跨模态注意力机制、特征加权融合和自适应权重分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, seq_dim: int, struct_dim: int, hidden_dim: int):
        super().__init__()
        self.seq_dim = seq_dim
        self.struct_dim = struct_dim 
        self.hidden_dim = hidden_dim
        
        # 序列特征投影
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        
        # 跨模态注意力
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        self.value_net = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, seq_features: torch.Tensor, struct_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """跨模态注意力计算"""
        # 特征投影
        seq_proj = self.seq_proj(seq_features)  # [batch, seq_len, hidden_dim]
        struct_proj = self.struct_proj(struct_features)  # [batch, struct_len, hidden_dim]
        
        # 计算注意力权重
        query = self.query_net(seq_proj)
        key = self.key_net(struct_proj) 
        value = self.value_net(struct_proj)
        
        # 注意力计算
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch, seq_len, struct_len]
        attention_weights = F.softmax(attention_scores / np.sqrt(self.hidden_dim), dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权特征融合
        attended_features = torch.bmm(attention_weights, value)  # [batch, seq_len, hidden_dim]
        
        # 残差连接和归一化
        fused_features = self.norm(seq_proj + attended_features)
        
        return fused_features, attention_weights

class AdaptiveWeightFusion(nn.Module):
    """自适应权重融合模块"""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.num_modalities = len(input_dims)
        
        # 特征投影网络
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        # 权重生成网络
        total_dim = sum(input_dims)
        self.weight_net = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Linear(total_dim // 2, self.num_modalities),
            nn.Softmax(dim=-1)
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """自适应权重融合"""
        batch_size = features[0].shape[0]
        
        # 投影所有特征到相同维度
        projected_features = []
        for i, feature in enumerate(features):
            projected = self.projections[i](feature)
            projected_features.append(projected)
            
        # 计算自适应权重
        concat_features = torch.cat([f.mean(dim=1) if f.dim() > 2 else f for f in features], dim=-1)
        weights = self.weight_net(concat_features)  # [batch, num_modalities]
        
        # 加权融合
        weighted_sum = torch.zeros_like(projected_features[0])
        for i, feature in enumerate(projected_features):
            if feature.dim() == 3:  # 序列特征
                weight = weights[:, i].unsqueeze(1).unsqueeze(2)
            else:  # 标量特征
                weight = weights[:, i].unsqueeze(1)
            weighted_sum += weight * feature
            
        # 融合处理
        fused_output = self.fusion_net(weighted_sum)
        
        return fused_output, weights

class MultiModalFusionNetwork(nn.Module):
    """多模态特征融合网络"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 配置参数
        self.seq_dim = config.get('seq_dim', 512)
        self.struct_dim = config.get('struct_dim', 256)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.output_dim = config.get('output_dim', 256)
        self.use_cross_attention = config.get('use_cross_attention', True)
        
        # 跨模态注意力
        if self.use_cross_attention:
            self.cross_attention = CrossModalAttention(
                self.seq_dim, self.struct_dim, self.hidden_dim
            )
            input_dims = [self.hidden_dim, self.struct_dim]
        else:
            input_dims = [self.seq_dim, self.struct_dim]
            
        # 自适应权重融合
        self.adaptive_fusion = AdaptiveWeightFusion(input_dims, self.output_dim)
        
        # 输出层
        self.output_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
    def forward(self, 
                seq_features: torch.Tensor,
                struct_features: torch.Tensor,
                additional_features: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """多模态特征融合前向传播"""
        
        results = {}
        
        # 跨模态注意力处理
        if self.use_cross_attention:
            fused_seq_features, attention_weights = self.cross_attention(seq_features, struct_features)
            features_list = [fused_seq_features, struct_features]
            results['attention_weights'] = attention_weights
        else:
            features_list = [seq_features, struct_features]
            
        # 添加额外特征
        if additional_features:
            features_list.extend(additional_features)
            
        # 自适应权重融合
        fused_output, fusion_weights = self.adaptive_fusion(features_list)
        results['fusion_weights'] = fusion_weights
        
        # 最终输出处理
        final_output = self.output_head(fused_output)
        results['fused_features'] = final_output
        
        return results

class FeatureExtractor(nn.Module):
    """特征提取器"""
    
    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int):
        super().__init__()
        
        # 序列嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=6)
        
        # 结构特征提取
        self.struct_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256)
        )
        
    def forward(self, sequences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """特征提取"""
        batch_size, seq_len = sequences.shape
        
        # 序列特征提取
        seq_embed = self.embedding(sequences)
        pos_ids = torch.arange(seq_len, device=sequences.device).expand(batch_size, -1)
        pos_embed = self.pos_embedding(pos_ids)
        
        seq_features = self.transformer(seq_embed + pos_embed)
        
        # 结构特征提取（简化版本）
        struct_input = sequences.float().unsqueeze(1) / sequences.max()
        struct_features = self.struct_extractor(struct_input).transpose(1, 2)
        
        return {
            'sequence_features': seq_features,
            'structure_features': struct_features
        }

class MultiModalPredictor(nn.Module):
    """多模态启动子强度预测器"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        vocab_size = config.get('vocab_size', 5)  # A, T, G, C, N
        seq_len = config.get('seq_len', 1000)
        embed_dim = config.get('embed_dim', 512)
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(vocab_size, seq_len, embed_dim)
        
        # 多模态融合网络
        fusion_config = {
            'seq_dim': embed_dim,
            'struct_dim': 256,
            'hidden_dim': 512,
            'output_dim': 256,
            'use_cross_attention': True
        }
        self.fusion_network = MultiModalFusionNetwork(fusion_config)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sequences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """多模态预测前向传播"""
        
        # 特征提取
        features = self.feature_extractor(sequences)
        
        # 多模态融合
        fusion_results = self.fusion_network(
            features['sequence_features'],
            features['structure_features']
        )
        
        # 强度预测
        fused_features = fusion_results['fused_features']
        if fused_features.dim() == 3:
            pooled_features = torch.mean(fused_features, dim=1)
        else:
            pooled_features = fused_features
            
        predictions = self.prediction_head(pooled_features)
        
        return {
            'predictions': predictions,
            'fused_features': fused_features,
            'attention_weights': fusion_results.get('attention_weights'),
            'fusion_weights': fusion_results.get('fusion_weights')
        }

def create_multimodal_fusion_model(config: Dict) -> MultiModalPredictor:
    """创建多模态融合模型"""
    model = MultiModalPredictor(config)
    return model

if __name__ == "__main__":
    # 测试示例
    config = {
        'vocab_size': 5,
        'seq_len': 1000,
        'embed_dim': 512
    }
    
    model = create_multimodal_fusion_model(config)
    
    # 测试数据
    batch_size = 8
    seq_len = 1000
    test_sequences = torch.randint(0, 5, (batch_size, seq_len))
    
    # 前向传播测试
    with torch.no_grad():
        results = model(test_sequences)
        print(f"预测结果形状: {results['predictions'].shape}")
        print(f"融合特征形状: {results['fused_features'].shape}")
        if results['attention_weights'] is not None:
            print(f"注意力权重形状: {results['attention_weights'].shape}")
        print(f"融合权重形状: {results['fusion_weights'].shape}")
