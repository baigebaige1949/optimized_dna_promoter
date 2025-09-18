"""改进的特征提取器"""

import torch
import torch.nn as nn
import numpy as np
import itertools
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from ..config.model_config import PredictorModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BiologicalFeatureExtractor:
    """生物学特征提取器"""
    
    def __init__(self):
        # 常见的转录因子结合位点motif
        self.motifs = {
            'TATA': ['TATAAA', 'TATAWA', 'TAWAWW'],  # W = A or T
            'CAAT': ['CCAAT', 'CAATCT'],
            'GC': ['GGGCGG', 'GGGGCG'],
            'E_box': ['CANNTG'],  # N = any nucleotide
            'NF_kB': ['GGGACTTTCC', 'GGRACTTTCC']  # R = A or G
        }
        
        # 常见的CpG岛特征
        self.cpg_threshold = 0.6  # CpG观察/期望值比值
        
        logger.info("Initialized biological feature extractor")
    
    def extract_gc_content(self, sequence: str) -> float:
        """计算GC含量"""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0
    
    def extract_cpg_features(self, sequence: str) -> Dict[str, float]:
        """提取CpG相关特征"""
        features = {}
        
        # CpG二核苷酸密度
        cpg_count = sequence.count('CG')
        features['cpg_density'] = cpg_count / (len(sequence) - 1) if len(sequence) > 1 else 0.0
        
        # CpG观察/期望值比值
        c_count = sequence.count('C')
        g_count = sequence.count('G')
        expected_cpg = (c_count * g_count) / len(sequence) if len(sequence) > 0 else 0.0
        
        if expected_cpg > 0:
            features['cpg_oe_ratio'] = cpg_count / expected_cpg
        else:
            features['cpg_oe_ratio'] = 0.0
        
        # CpG岛预测
        features['is_cpg_island'] = (
            features['cpg_oe_ratio'] > self.cpg_threshold and 
            features['cpg_density'] > 0.05 and
            len(sequence) >= 200
        )
        
        return features
    
    def extract_motif_features(self, sequence: str) -> Dict[str, Any]:
        """提取motif特征"""
        features = {}
        found_motifs = []
        
        sequence_upper = sequence.upper()
        
        for motif_type, patterns in self.motifs.items():
            motif_found = False
            motif_positions = []
            
            for pattern in patterns:
                # 处理正则表达式样式
                if 'W' in pattern:  # W = A or T
                    pattern_variants = [pattern.replace('W', 'A'), pattern.replace('W', 'T')]
                elif 'R' in pattern:  # R = A or G
                    pattern_variants = [pattern.replace('R', 'A'), pattern.replace('R', 'G')]
                elif 'N' in pattern:  # N = any nucleotide
                    # 简化处理，只检测最常见的变体
                    pattern_variants = [pattern.replace('N', n) for n in ['A', 'T', 'G', 'C']]
                else:
                    pattern_variants = [pattern]
                
                for variant in pattern_variants:
                    if variant in sequence_upper:
                        motif_found = True
                        # 找到所有位置
                        start = 0
                        while True:
                            pos = sequence_upper.find(variant, start)
                            if pos == -1:
                                break
                            motif_positions.append(pos)
                            start = pos + 1
            
            features[f'{motif_type}_found'] = motif_found
            features[f'{motif_type}_count'] = len(motif_positions)
            
            if motif_found:
                found_motifs.append(motif_type)
                # 计算相对位置（在序列的前30%、中间40%、后30%）
                positions_normalized = [pos / len(sequence) for pos in motif_positions]
                features[f'{motif_type}_avg_position'] = np.mean(positions_normalized) if positions_normalized else 0.0
        
        features['total_motifs_found'] = len(found_motifs)
        features['motif_types'] = found_motifs
        
        return features
    
    def extract_nucleotide_composition(self, sequence: str) -> Dict[str, float]:
        """提取核苷酸组成特征"""
        sequence_upper = sequence.upper()
        length = len(sequence_upper)
        
        features = {}
        
        # 单核苷酸频率
        for nucleotide in ['A', 'T', 'G', 'C']:
            features[f'{nucleotide}_content'] = sequence_upper.count(nucleotide) / length if length > 0 else 0.0
        
        # AT/GC比值
        at_content = features['A_content'] + features['T_content']
        gc_content = features['G_content'] + features['C_content']
        
        features['at_gc_ratio'] = at_content / gc_content if gc_content > 0 else float('inf')
        features['purine_content'] = features['A_content'] + features['G_content']  # 嘉唶呂
        features['pyrimidine_content'] = features['T_content'] + features['C_content']  # 嘋唶定
        
        return features
    
    def extract_dinucleotide_features(self, sequence: str) -> Dict[str, float]:
        """提取二核苷酸特征"""
        sequence_upper = sequence.upper()
        dinucleotides = []
        
        # 提取所有二核苷酸
        for i in range(len(sequence_upper) - 1):
            dinucleotides.append(sequence_upper[i:i+2])
        
        # 计算频率
        dinuc_counts = Counter(dinucleotides)
        total_dinucs = len(dinucleotides)
        
        features = {}
        nucleotides = ['A', 'T', 'G', 'C']
        
        for n1 in nucleotides:
            for n2 in nucleotides:
                dinuc = n1 + n2
                features[f'{dinuc}_freq'] = dinuc_counts.get(dinuc, 0) / total_dinucs if total_dinucs > 0 else 0.0
        
        return features
    
    def extract_all_features(self, sequence: str) -> Dict[str, Any]:
        """提取所有生物学特征"""
        features = {}
        
        # 基本特征
        features.update(self.extract_nucleotide_composition(sequence))
        features.update(self.extract_cpg_features(sequence))
        features.update(self.extract_motif_features(sequence))
        features.update(self.extract_dinucleotide_features(sequence))
        
        # 序列长度
        features['sequence_length'] = len(sequence)
        
        return features


class KmerExtractor:
    """
k-mer特征提取器
"""
    
    def __init__(self, kmer_sizes: List[int] = [3, 4, 5]):
        self.kmer_sizes = kmer_sizes
        self.nucleotides = ['A', 'T', 'G', 'C']
        
        # 预计算所有可能的k-mer
        self.all_kmers = {}
        for k in kmer_sizes:
            self.all_kmers[k] = [''.join(p) for p in itertools.product(self.nucleotides, repeat=k)]
        
        logger.info(f"Initialized k-mer extractor with k sizes: {kmer_sizes}")
    
    def extract_kmer_features(self, sequence: str, k: int) -> np.ndarray:
        """提取特定k值的k-mer特征"""
        sequence_upper = sequence.upper()
        
        # 提取k-mer
        kmers = []
        for i in range(len(sequence_upper) - k + 1):
            kmers.append(sequence_upper[i:i+k])
        
        # 计算频率
        kmer_counts = Counter(kmers)
        total_kmers = len(kmers)
        
        # 创建特征向量
        features = []
        for kmer in self.all_kmers[k]:
            freq = kmer_counts.get(kmer, 0) / total_kmers if total_kmers > 0 else 0.0
            features.append(freq)
        
        return np.array(features)
    
    def extract_all_kmer_features(self, sequence: str) -> Dict[str, np.ndarray]:
        """提取所有k值的k-mer特征"""
        features = {}
        for k in self.kmer_sizes:
            features[f'kmer_{k}'] = self.extract_kmer_features(sequence, k)
        return features


class FeatureExtractor(nn.Module):
    """主特征提取器，结合多种特征提取方法"""
    
    def __init__(self, config: PredictorModelConfig):
        super().__init__()
        self.config = config
        
        # 初始化子组件
        self.biological_extractor = BiologicalFeatureExtractor()
        self.kmer_extractor = KmerExtractor(config.kmer_sizes)
        
        # 计算特征维度
        self.feature_dims = self._calculate_feature_dims()
        
        # 特征融合层
        if config.feature_fusion_method == "concat":
            self.fusion_layer = nn.Identity()
            self.output_dim = sum(self.feature_dims.values())
        elif config.feature_fusion_method == "attention":
            self.fusion_layer = AttentionFusion(self.feature_dims)
            self.output_dim = config.classifier_hidden_dim
        else:
            raise ValueError(f"Unknown fusion method: {config.feature_fusion_method}")
        
        logger.info(f"Initialized feature extractor with output dim: {self.output_dim}")
    
    def _calculate_feature_dims(self) -> Dict[str, int]:
        """计算各类特征的维度"""
        dims = {}
        
        # k-mer特征维度
        for k in self.config.kmer_sizes:
            dims[f'kmer_{k}'] = 4 ** k
        
        # 生物学特征维度（估算）
        if self.config.use_biological_features:
            dims['biological'] = 50  # 估计值，包括各种生物学特征
        
        return dims
    
    def extract_features(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """提取批量序列的特征"""
        batch_size = len(sequences)
        features = {}
        
        # 提取k-mer特征
        for k in self.config.kmer_sizes:
            kmer_features = []
            for seq in sequences:
                kmer_feat = self.kmer_extractor.extract_kmer_features(seq, k)
                kmer_features.append(kmer_feat)
            features[f'kmer_{k}'] = torch.tensor(np.stack(kmer_features), dtype=torch.float32)
        
        # 提取生物学特征
        if self.config.use_biological_features:
            bio_features = []
            for seq in sequences:
                bio_feat = self.biological_extractor.extract_all_features(seq)
                # 转换为数值向量
                bio_vector = self._bio_features_to_vector(bio_feat)
                bio_features.append(bio_vector)
            features['biological'] = torch.tensor(np.stack(bio_features), dtype=torch.float32)
        
        return features
    
    def _bio_features_to_vector(self, bio_features: Dict[str, Any]) -> np.ndarray:
        """将生物学特征字典转换为数值向量"""
        vector = []
        
        # 核苷酸组成特征
        for nucleotide in ['A', 'T', 'G', 'C']:
            vector.append(bio_features.get(f'{nucleotide}_content', 0.0))
        
        # CpG特征
        vector.append(bio_features.get('cpg_density', 0.0))
        vector.append(bio_features.get('cpg_oe_ratio', 0.0))
        vector.append(float(bio_features.get('is_cpg_island', False)))
        
        # Motif特征
        motif_types = ['TATA', 'CAAT', 'GC', 'E_box', 'NF_kB']
        for motif in motif_types:
            vector.append(float(bio_features.get(f'{motif}_found', False)))
            vector.append(bio_features.get(f'{motif}_count', 0))
        
        # 其他特征
        vector.append(bio_features.get('at_gc_ratio', 0.0))
        vector.append(bio_features.get('purine_content', 0.0))
        vector.append(bio_features.get('pyrimidine_content', 0.0))
        
        # 二核苷酸特征（只取主要的几个）
        important_dinucs = ['CG', 'GC', 'AT', 'TA', 'AA', 'TT', 'GG', 'CC']
        for dinuc in important_dinucs:
            vector.append(bio_features.get(f'{dinuc}_freq', 0.0))
        
        # 补齐到固定长度
        while len(vector) < 50:
            vector.append(0.0)
        
        return np.array(vector[:50])  # 保证固定长度
    
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """前向传播"""
        features = self.extract_features(sequences)
        
        if self.config.feature_fusion_method == "concat":
            # 简单拼接
            feature_list = []
            for k in self.config.kmer_sizes:
                feature_list.append(features[f'kmer_{k}'])
            if self.config.use_biological_features:
                feature_list.append(features['biological'])
            
            fused_features = torch.cat(feature_list, dim=1)
        else:
            # 注意力融合
            fused_features = self.fusion_layer(features)
        
        return fused_features


class AttentionFusion(nn.Module):
    """基于注意力的特征融合"""
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dim: int = 256):
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # 为每种特征创建投射层
        self.projections = nn.ModuleDict()
        for feat_name, dim in feature_dims.items():
            self.projections[feat_name] = nn.Linear(dim, hidden_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # 输出层
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = list(features.values())[0].size(0)
        
        # 投射所有特征到统一维度
        projected_features = []
        for feat_name, feat_tensor in features.items():
            projected = self.projections[feat_name](feat_tensor)
            projected_features.append(projected.unsqueeze(1))  # [batch, 1, hidden_dim]
        
        # 拼接所有特征
        all_features = torch.cat(projected_features, dim=1)  # [batch, num_features, hidden_dim]
        
        # 应用注意力
        attended_features, _ = self.attention(all_features, all_features, all_features)
        
        # 池化（取平均值）
        pooled_features = attended_features.mean(dim=1)  # [batch, hidden_dim]
        
        # 输出投射
        output = self.output_projection(pooled_features)
        
        return output
