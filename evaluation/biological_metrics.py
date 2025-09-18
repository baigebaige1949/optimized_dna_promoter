# -*- coding: utf-8 -*-
"""生物学评估指标模块

实现Jensen-Shannon散度、S-FID、Motif分析、GC含量、序列相似度等生物学指标
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import pairwise_distances
from collections import Counter, defaultdict
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import GC
from Bio.Align import PairwiseAligner
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class BiologicalMetrics:
    """生物学评估指标类"""
    
    def __init__(self, nucleotides: List[str] = None):
        self.nucleotides = nucleotides or ['A', 'T', 'G', 'C']
        self.nucleotide_to_idx = {nt: i for i, nt in enumerate(self.nucleotides)}
        
        # 常见启动子motifs (TATA box, CpG island等)
        self.common_motifs = {
            'TATA_box': ['TATAAA', 'TATAWA', 'TATAWR'],  # W = A or T, R = A or G
            'CAAT_box': ['CAAT', 'CCAAT'],
            'GC_box': ['GGGCGG', 'CCGCCC'],
            'CpG_site': ['CG'],
            'Initiator': ['YYANWYY'],  # Y = C or T, N = any, W = A or T
            'DPE': ['RGWCGTG'],  # R = A or G, W = A or T
        }
        
        # 配置PairwiseAligner
        self.aligner = PairwiseAligner()
        self.aligner.match_score = 1
        self.aligner.mismatch_score = -1
        self.aligner.open_gap_score = -2
        self.aligner.extend_gap_score = -0.5
        
    def sequences_to_distributions(self, sequences: List[str], k: int = 1) -> np.ndarray:
        """将序列转换为k-mer分布"""
        if k == 1:
            # 单核苷酸分布
            distributions = []
            for seq in sequences:
                counts = Counter(seq.upper())
                dist = np.array([counts.get(nt, 0) for nt in self.nucleotides])
                dist = dist / dist.sum() if dist.sum() > 0 else dist
                distributions.append(dist)
            return np.array(distributions)
        else:
            # k-mer分布
            all_kmers = [''.join(p) for p in self._generate_kmers(k)]
            kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
            
            distributions = []
            for seq in sequences:
                seq = seq.upper()
                kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
                counts = Counter(kmers)
                dist = np.array([counts.get(kmer, 0) for kmer in all_kmers])
                dist = dist / dist.sum() if dist.sum() > 0 else dist
                distributions.append(dist)
            return np.array(distributions)
            
    def _generate_kmers(self, k: int) -> List[Tuple[str, ...]]:
        """生成所有可能的k-mer"""
        if k == 1:
            return [(nt,) for nt in self.nucleotides]
        else:
            smaller_kmers = self._generate_kmers(k-1)
            return [(nt,) + kmer for nt in self.nucleotides for kmer in smaller_kmers]
            
    def jensen_shannon_divergence(self, 
                                real_sequences: List[str], 
                                generated_sequences: List[str],
                                k: int = 1) -> float:
        """计算Jensen-Shannon散度"""
        
        # 获取k-mer分布
        real_dists = self.sequences_to_distributions(real_sequences, k)
        gen_dists = self.sequences_to_distributions(generated_sequences, k)
        
        # 计算平均分布
        real_mean_dist = np.mean(real_dists, axis=0)
        gen_mean_dist = np.mean(gen_dists, axis=0)
        
        # 添加小的平滑项避免零概率
        epsilon = 1e-10
        real_mean_dist = real_mean_dist + epsilon
        gen_mean_dist = gen_mean_dist + epsilon
        
        # 归一化
        real_mean_dist = real_mean_dist / real_mean_dist.sum()
        gen_mean_dist = gen_mean_dist / gen_mean_dist.sum()
        
        # 计算JS散度
        js_div = jensenshannon(real_mean_dist, gen_mean_dist) ** 2
        return js_div
        
    def sequence_fid(self, 
                    real_sequences: List[str], 
                    generated_sequences: List[str],
                    feature_dim: int = 100) -> float:
        """序列FID (S-FID) 计算"""
        
        # 提取序列特征
        real_features = self._extract_sequence_features(real_sequences, feature_dim)
        gen_features = self._extract_sequence_features(generated_sequences, feature_dim)
        
        # 计算FID
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(gen_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # 计算FID分数
        diff = mu_real - mu_gen
        
        # 计算矩阵平方根
        from scipy.linalg import sqrtm
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        
        # 处理数值误差
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = np.sum(diff**2) + np.trace(sigma_real + sigma_gen - 2*covmean)
        return float(fid)
        
    def _extract_sequence_features(self, sequences: List[str], feature_dim: int) -> np.ndarray:
        """提取序列特征用于FID计算"""
        features = []
        
        for seq in sequences:
            seq = seq.upper()
            feature_vector = []
            
            # 1. 基本组成特征
            for nt in self.nucleotides:
                feature_vector.append(seq.count(nt) / len(seq))
                
            # 2. 二核苷酸频率
            dinucleotides = [seq[i:i+2] for i in range(len(seq)-1)]
            dinuc_counts = Counter(dinucleotides)
            total_dinuc = len(dinucleotides)
            
            for nt1 in self.nucleotides:
                for nt2 in self.nucleotides:
                    dinuc = nt1 + nt2
                    feature_vector.append(dinuc_counts.get(dinuc, 0) / total_dinuc if total_dinuc > 0 else 0)
                    
            # 3. GC含量
            feature_vector.append(self.gc_content([seq])[0])
            
            # 4. 序列复杂度（Shannon熵）
            feature_vector.append(self._sequence_entropy(seq))
            
            # 5. Motif数量
            motif_counts = self.count_motifs([seq])
            for motif_type in self.common_motifs:
                feature_vector.append(motif_counts[motif_type][0] / len(seq))
                
            # 截断或补齐到指定维度
            if len(feature_vector) > feature_dim:
                feature_vector = feature_vector[:feature_dim]
            else:
                feature_vector.extend([0] * (feature_dim - len(feature_vector)))
                
            features.append(feature_vector)
            
        return np.array(features)
        
    def _sequence_entropy(self, sequence: str) -> float:
        """计算序列的Shannon熵"""
        counts = Counter(sequence.upper())
        total = sum(counts.values())
        
        if total == 0:
            return 0.0
            
        probabilities = [count / total for count in counts.values()]
        return entropy(probabilities, base=2)
        
    def gc_content(self, sequences: List[str]) -> List[float]:
        """计算GC含量"""
        gc_contents = []
        for seq in sequences:
            seq = seq.upper()
            gc_count = seq.count('G') + seq.count('C')
            total_count = len(seq)
            gc_contents.append(gc_count / total_count if total_count > 0 else 0.0)
        return gc_contents
        
    def count_motifs(self, sequences: List[str]) -> Dict[str, List[int]]:
        """计算motif数量"""
        motif_counts = {motif_type: [] for motif_type in self.common_motifs}
        
        for seq in sequences:
            seq = seq.upper()
            for motif_type, motif_patterns in self.common_motifs.items():
                total_count = 0
                for pattern in motif_patterns:
                    # 处理IUPAC核苷酸代码
                    regex_pattern = self._iupac_to_regex(pattern)
                    matches = re.findall(regex_pattern, seq)
                    total_count += len(matches)
                motif_counts[motif_type].append(total_count)
                
        return motif_counts
        
    def _iupac_to_regex(self, iupac_pattern: str) -> str:
        """将IUPAC核苷酸代码转换为正则表达式"""
        iupac_map = {
            'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C',
            'R': '[AG]',    # A or G
            'Y': '[CT]',    # C or T
            'S': '[GC]',    # G or C
            'W': '[AT]',    # A or T
            'K': '[GT]',    # G or T
            'M': '[AC]',    # A or C
            'B': '[CGT]',   # C or G or T
            'D': '[AGT]',   # A or G or T
            'H': '[ACT]',   # A or C or T
            'V': '[ACG]',   # A or C or G
            'N': '[ATGC]'   # any
        }
        
        regex_pattern = ''
        for char in iupac_pattern:
            regex_pattern += iupac_map.get(char, char)
            
        return regex_pattern
        
    def sequence_similarity(self, 
                          sequences1: List[str], 
                          sequences2: List[str],
                          method: str = 'alignment') -> Dict[str, float]:
        """计算序列相似度"""
        
        if method == 'alignment':
            return self._alignment_similarity(sequences1, sequences2)
        elif method == 'kmer':
            return self._kmer_similarity(sequences1, sequences2)
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")
            
    def _alignment_similarity(self, 
                            sequences1: List[str], 
                            sequences2: List[str]) -> Dict[str, float]:
        """基于序列比对的相似度计算"""
        similarities = []
        
        # 采样策略：如果序列太多，随机采样
        max_comparisons = 1000
        if len(sequences1) * len(sequences2) > max_comparisons:
            sample_size = int(np.sqrt(max_comparisons))
            sequences1 = np.random.choice(sequences1, min(sample_size, len(sequences1)), replace=False)
            sequences2 = np.random.choice(sequences2, min(sample_size, len(sequences2)), replace=False)
            
        for seq1 in sequences1:
            for seq2 in sequences2:
                try:
                    alignments = self.aligner.align(seq1, seq2)
                    if alignments:
                        best_alignment = alignments[0]
                        similarity = best_alignment.score / max(len(seq1), len(seq2))
                        similarities.append(similarity)
                except Exception as e:
                    # 如果比对失败，使用简单的相似度计算
                    similarity = self._simple_similarity(seq1, seq2)
                    similarities.append(similarity)
                    
        return {
            'mean_similarity': np.mean(similarities) if similarities else 0.0,
            'std_similarity': np.std(similarities) if similarities else 0.0,
            'max_similarity': np.max(similarities) if similarities else 0.0,
            'min_similarity': np.min(similarities) if similarities else 0.0
        }
        
    def _simple_similarity(self, seq1: str, seq2: str) -> float:
        """简单的序列相似度计算"""
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
            
        matches = sum(c1 == c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
        return matches / min_len
        
    def _kmer_similarity(self, 
                       sequences1: List[str], 
                       sequences2: List[str], 
                       k: int = 3) -> Dict[str, float]:
        """基于k-mer的相似度计算"""
        
        # 获取k-mer分布
        dists1 = self.sequences_to_distributions(sequences1, k)
        dists2 = self.sequences_to_distributions(sequences2, k)
        
        # 计算平均分布
        mean_dist1 = np.mean(dists1, axis=0)
        mean_dist2 = np.mean(dists2, axis=0)
        
        # 计算相似度指标
        cosine_sim = np.dot(mean_dist1, mean_dist2) / (np.linalg.norm(mean_dist1) * np.linalg.norm(mean_dist2))
        pearson_corr = np.corrcoef(mean_dist1, mean_dist2)[0, 1]
        js_distance = jensenshannon(mean_dist1, mean_dist2)
        
        return {
            'cosine_similarity': float(cosine_sim),
            'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
            'js_distance': float(js_distance),
            'js_similarity': 1 - float(js_distance)
        }
        
    def motif_enrichment_analysis(self, 
                                sequences: List[str], 
                                background_sequences: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Motif富集分析"""
        
        # 计算目标序列中的motif数量
        target_motifs = self.count_motifs(sequences)
        
        # 如果没有背景序列，生成随机背景
        if background_sequences is None:
            background_sequences = self._generate_random_sequences(
                len(sequences), 
                np.mean([len(seq) for seq in sequences])
            )
            
        background_motifs = self.count_motifs(background_sequences)
        
        enrichment_results = {}
        
        for motif_type in self.common_motifs:
            target_counts = np.array(target_motifs[motif_type])
            background_counts = np.array(background_motifs[motif_type])
            
            target_rate = np.mean(target_counts) / np.mean([len(seq) for seq in sequences])
            background_rate = np.mean(background_counts) / np.mean([len(seq) for seq in background_sequences])
            
            enrichment_fold = target_rate / background_rate if background_rate > 0 else float('inf')
            
            # 统计显著性检验
            try:
                from scipy.stats import mannwhitneyu
                statistic, p_value = mannwhitneyu(target_counts, background_counts, alternative='greater')
            except:
                p_value = 1.0
                
            enrichment_results[motif_type] = {
                'target_rate': target_rate,
                'background_rate': background_rate,
                'enrichment_fold': enrichment_fold,
                'p_value': p_value,
                'is_enriched': enrichment_fold > 1.5 and p_value < 0.05
            }
            
        return enrichment_results
        
    def _generate_random_sequences(self, num_sequences: int, avg_length: float) -> List[str]:
        """生成随机DNA序列作为背景"""
        random_sequences = []
        
        for _ in range(num_sequences):
            length = int(np.random.normal(avg_length, avg_length * 0.1))
            length = max(50, min(2000, length))  # 限制长度范围
            
            sequence = ''.join(np.random.choice(self.nucleotides, length))
            random_sequences.append(sequence)
            
        return random_sequences
        
    def comprehensive_evaluation(self, 
                               real_sequences: List[str], 
                               generated_sequences: List[str],
                               save_plots: bool = True,
                               output_dir: str = 'evaluation_results') -> Dict[str, Any]:
        """综合生物学评估"""
        
        results = {}
        
        # 1. Jensen-Shannon散度
        print("计算Jensen-Shannon散度...")
        results['js_divergence_1mer'] = self.jensen_shannon_divergence(
            real_sequences, generated_sequences, k=1
        )
        results['js_divergence_2mer'] = self.jensen_shannon_divergence(
            real_sequences, generated_sequences, k=2
        )
        results['js_divergence_3mer'] = self.jensen_shannon_divergence(
            real_sequences, generated_sequences, k=3
        )
        
        # 2. S-FID
        print("计算S-FID...")
        results['s_fid'] = self.sequence_fid(real_sequences, generated_sequences)
        
        # 3. GC含量分析
        print("分析GC含量...")
        real_gc = self.gc_content(real_sequences)
        gen_gc = self.gc_content(generated_sequences)
        
        results['gc_content'] = {
            'real_mean': np.mean(real_gc),
            'real_std': np.std(real_gc),
            'generated_mean': np.mean(gen_gc),
            'generated_std': np.std(gen_gc),
            'difference': abs(np.mean(real_gc) - np.mean(gen_gc))
        }
        
        # 4. 序列长度分析
        real_lengths = [len(seq) for seq in real_sequences]
        gen_lengths = [len(seq) for seq in generated_sequences]
        
        results['sequence_length'] = {
            'real_mean': np.mean(real_lengths),
            'real_std': np.std(real_lengths),
            'generated_mean': np.mean(gen_lengths),
            'generated_std': np.std(gen_lengths),
            'difference': abs(np.mean(real_lengths) - np.mean(gen_lengths))
        }
        
        # 5. Motif分析
        print("分析Motif富集...")
        real_motifs = self.count_motifs(real_sequences)
        gen_motifs = self.count_motifs(generated_sequences)
        
        motif_analysis = {}
        for motif_type in self.common_motifs:
            real_counts = np.array(real_motifs[motif_type])
            gen_counts = np.array(gen_motifs[motif_type])
            
            motif_analysis[motif_type] = {
                'real_mean': np.mean(real_counts),
                'real_std': np.std(real_counts),
                'generated_mean': np.mean(gen_counts),
                'generated_std': np.std(gen_counts),
                'difference': abs(np.mean(real_counts) - np.mean(gen_counts))
            }
            
        results['motif_analysis'] = motif_analysis
        
        # 6. 序列相似度
        print("计算序列相似度...")
        similarity_results = self.sequence_similarity(
            real_sequences[:100],  # 采样以提高速度
            generated_sequences[:100],
            method='kmer'
        )
        results['sequence_similarity'] = similarity_results
        
        # 7. 多样性分析
        results['diversity'] = {
            'real_unique_ratio': len(set(real_sequences)) / len(real_sequences),
            'generated_unique_ratio': len(set(generated_sequences)) / len(generated_sequences)
        }
        
        # 保存可视化结果
        if save_plots:
            self._create_evaluation_plots(
                real_sequences, generated_sequences, results, output_dir
            )
            
        return results
        
    def _create_evaluation_plots(self, 
                               real_sequences: List[str], 
                               generated_sequences: List[str],
                               results: Dict[str, Any],
                               output_dir: str):
        """创建评估可视化图表"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # 1. GC含量分布比较
        plt.figure(figsize=(12, 8))
        
        real_gc = self.gc_content(real_sequences)
        gen_gc = self.gc_content(generated_sequences)
        
        plt.subplot(2, 2, 1)
        plt.hist(real_gc, bins=30, alpha=0.7, label='真实序列', density=True)
        plt.hist(gen_gc, bins=30, alpha=0.7, label='生成序列', density=True)
        plt.xlabel('GC含量')
        plt.ylabel('密度')
        plt.title('GC含量分布比较')
        plt.legend()
        
        # 2. 序列长度分布比较
        real_lengths = [len(seq) for seq in real_sequences]
        gen_lengths = [len(seq) for seq in generated_sequences]
        
        plt.subplot(2, 2, 2)
        plt.hist(real_lengths, bins=30, alpha=0.7, label='真实序列', density=True)
        plt.hist(gen_lengths, bins=30, alpha=0.7, label='生成序列', density=True)
        plt.xlabel('序列长度')
        plt.ylabel('密度')
        plt.title('序列长度分布比较')
        plt.legend()
        
        # 3. Motif数量比较
        real_motifs = self.count_motifs(real_sequences)
        gen_motifs = self.count_motifs(generated_sequences)
        
        plt.subplot(2, 2, 3)
        motif_types = list(self.common_motifs.keys())
        real_means = [np.mean(real_motifs[mt]) for mt in motif_types]
        gen_means = [np.mean(gen_motifs[mt]) for mt in motif_types]
        
        x = np.arange(len(motif_types))
        width = 0.35
        
        plt.bar(x - width/2, real_means, width, label='真实序列', alpha=0.8)
        plt.bar(x + width/2, gen_means, width, label='生成序列', alpha=0.8)
        
        plt.xlabel('Motif类型')
        plt.ylabel('平均数量')
        plt.title('Motif数量比较')
        plt.xticks(x, motif_types, rotation=45)
        plt.legend()
        
        # 4. JS散度和FID分数
        plt.subplot(2, 2, 4)
        metrics = ['JS-1mer', 'JS-2mer', 'JS-3mer', 'S-FID']
        scores = [
            results['js_divergence_1mer'],
            results['js_divergence_2mer'], 
            results['js_divergence_3mer'],
            results['s_fid'] / 1000  # 缩放FID分数便于显示
        ]
        
        plt.bar(metrics, scores, alpha=0.8)
        plt.ylabel('分数')
        plt.title('综合评估指标')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'biological_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 核苷酸组成比较热图
        real_comp = self.sequences_to_distributions(real_sequences, k=1)
        gen_comp = self.sequences_to_distributions(generated_sequences, k=1)
        
        real_mean_comp = np.mean(real_comp, axis=0)
        gen_mean_comp = np.mean(gen_comp, axis=0)
        
        comp_data = np.array([real_mean_comp, gen_mean_comp])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(comp_data, 
                   xticklabels=self.nucleotides,
                   yticklabels=['真实序列', '生成序列'],
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues')
        plt.title('核苷酸组成比较')
        plt.savefig(output_path / 'nucleotide_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"评估图表已保存到: {output_path}")

def evaluate_generated_sequences(real_sequences: List[str], 
                               generated_sequences: List[str],
                               output_dir: str = 'evaluation_results') -> Dict[str, Any]:
    """便捷的序列评估函数"""
    
    evaluator = BiologicalMetrics()
    results = evaluator.comprehensive_evaluation(
        real_sequences, 
        generated_sequences,
        save_plots=True,
        output_dir=output_dir
    )
    
    # 保存结果到JSON文件
    import json
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    return results

if __name__ == "__main__":
    # 测试示例
    
    # 生成测试序列
    np.random.seed(42)
    
    # 真实序列（模拟）
    real_sequences = [
        ''.join(np.random.choice(['A', 'T', 'G', 'C'], 200)) for _ in range(100)
    ]
    
    # 生成序列（模拟，稍有不同的分布）
    generated_sequences = [
        ''.join(np.random.choice(['A', 'T', 'G', 'C'], 200, p=[0.3, 0.3, 0.2, 0.2])) 
        for _ in range(100)
    ]
    
    # 运行评估
    print("开始生物学评估...")
    results = evaluate_generated_sequences(
        real_sequences, 
        generated_sequences,
        output_dir='test_evaluation'
    )
    
    print("\n评估结果:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"  {k}: {v}")
                else:
                    print(f"  {k}: {v:.6f}")
        else:
            print(f"{key}: {value:.6f}")
