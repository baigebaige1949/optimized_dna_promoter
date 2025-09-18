#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强数据处理模块
支持多种数据格式、数据增强和质量检查
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import csv
import json
from Bio import SeqIO
from Bio.SeqUtils import GC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import re
import warnings
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DataFormat:
    """数据格式常量"""
    FASTA = 'fasta'
    CSV = 'csv'
    GENBANK = 'genbank'
    JSON = 'json'
    TSV = 'tsv'

class SequenceValidator:
    """序列验证器"""
    
    def __init__(self):
        self.dna_pattern = re.compile(r'^[ATCGN]+$', re.IGNORECASE)
        self.rna_pattern = re.compile(r'^[AUCGN]+$', re.IGNORECASE)
    
    def is_valid_dna(self, sequence: str) -> bool:
        """验证DNA序列"""
        return bool(self.dna_pattern.match(sequence.strip()))
    
    def is_valid_rna(self, sequence: str) -> bool:
        """验证RNA序列"""
        return bool(self.rna_pattern.match(sequence.strip()))
    
    def validate_sequence(self, sequence: str, seq_type: str = 'dna') -> Tuple[bool, str]:
        """验证序列并返回错误信息"""
        if not sequence or len(sequence.strip()) == 0:
            return False, "空序列"
        
        sequence = sequence.strip().upper()
        
        if seq_type.lower() == 'dna':
            if not self.is_valid_dna(sequence):
                invalid_chars = set(sequence) - set('ATCGN')
                return False, f"包含非法DNA字符: {invalid_chars}"
        elif seq_type.lower() == 'rna':
            if not self.is_valid_rna(sequence):
                invalid_chars = set(sequence) - set('AUCGN')
                return False, f"包含非法RNA字符: {invalid_chars}"
        
        return True, ""
    
    def calculate_quality_metrics(self, sequence: str) -> Dict[str, float]:
        """计算序列质量指标"""
        if not sequence:
            return {}
        
        sequence = sequence.upper()
        length = len(sequence)
        
        metrics = {
            'length': length,
            'gc_content': GC(sequence) / 100.0,
            'n_ratio': sequence.count('N') / length if length > 0 else 0,
            'complexity': self._calculate_complexity(sequence),
            'repeat_ratio': self._calculate_repeat_ratio(sequence)
        }
        
        return metrics
    
    def _calculate_complexity(self, sequence: str) -> float:
        """计算序列复杂度（基于熵）"""
        if not sequence:
            return 0.0
        
        # 计算每个核苷酸的频率
        counts = {}
        for base in sequence:
            counts[base] = counts.get(base, 0) + 1
        
        # 计算熵
        length = len(sequence)
        entropy = 0.0
        for count in counts.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # 标准化到0-1范围
        max_entropy = np.log2(4)  # 4种核苷酸的最大熵
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_repeat_ratio(self, sequence: str, window_size: int = 10) -> float:
        """计算重复序列比例"""
        if len(sequence) < window_size * 2:
            return 0.0
        
        kmers = set()
        repeats = 0
        
        for i in range(len(sequence) - window_size + 1):
            kmer = sequence[i:i + window_size]
            if kmer in kmers:
                repeats += 1
            else:
                kmers.add(kmer)
        
        total_kmers = len(sequence) - window_size + 1
        return repeats / total_kmers if total_kmers > 0 else 0.0

class DataAugmentor:
    """数据增强器"""
    
    def __init__(self, augmentation_prob: float = 0.1):
        self.augmentation_prob = augmentation_prob
        self.nucleotides = ['A', 'T', 'C', 'G']
    
    def mutate_sequence(self, sequence: str, mutation_rate: float = 0.05) -> str:
        """随机突变序列"""
        sequence = list(sequence.upper())
        for i in range(len(sequence)):
            if np.random.random() < mutation_rate and sequence[i] in self.nucleotides:
                # 随机替换为其他核苷酸
                available = [n for n in self.nucleotides if n != sequence[i]]
                sequence[i] = np.random.choice(available)
        
        return ''.join(sequence)
    
    def insert_deletion(self, sequence: str, indel_rate: float = 0.02) -> str:
        """插入或删除操作"""
        sequence = list(sequence)
        i = 0
        while i < len(sequence):
            if np.random.random() < indel_rate:
                if np.random.random() < 0.5 and len(sequence) > 50:  # 删除
                    sequence.pop(i)
                else:  # 插入
                    sequence.insert(i, np.random.choice(self.nucleotides))
                    i += 1
            i += 1
        
        return ''.join(sequence)
    
    def complement_reverse(self, sequence: str) -> str:
        """反向互补"""
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        sequence = sequence.upper()
        complement = ''.join(complement_map.get(base, base) for base in sequence)
        return complement[::-1]
    
    def augment_batch(self, sequences: List[str], 
                      labels: Optional[List] = None) -> Tuple[List[str], Optional[List]]:
        """批量数据增强"""
        augmented_sequences = []
        augmented_labels = [] if labels is not None else None
        
        for i, seq in enumerate(sequences):
            # 原始序列
            augmented_sequences.append(seq)
            if labels is not None:
                augmented_labels.append(labels[i])
            
            # 增强序列
            if np.random.random() < self.augmentation_prob:
                # 突变
                if np.random.random() < 0.4:
                    aug_seq = self.mutate_sequence(seq)
                    augmented_sequences.append(aug_seq)
                    if labels is not None:
                        augmented_labels.append(labels[i])
                
                # 插入删除
                if np.random.random() < 0.3:
                    aug_seq = self.insert_deletion(seq)
                    augmented_sequences.append(aug_seq)
                    if labels is not None:
                        augmented_labels.append(labels[i])
                
                # 反向互补
                if np.random.random() < 0.3:
                    aug_seq = self.complement_reverse(seq)
                    augmented_sequences.append(aug_seq)
                    if labels is not None:
                        augmented_labels.append(labels[i])
        
        return augmented_sequences, augmented_labels

class MultiFormatDataLoader:
    """多格式数据加载器"""
    
    def __init__(self, validator: Optional[SequenceValidator] = None):
        self.validator = validator or SequenceValidator()
        self.supported_formats = {DataFormat.FASTA, DataFormat.CSV, 
                                DataFormat.GENBANK, DataFormat.JSON, DataFormat.TSV}
    
    def detect_format(self, file_path: Union[str, Path]) -> str:
        """自动检测文件格式"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        format_map = {
            '.fasta': DataFormat.FASTA,
            '.fa': DataFormat.FASTA,
            '.fas': DataFormat.FASTA,
            '.csv': DataFormat.CSV,
            '.tsv': DataFormat.TSV,
            '.txt': DataFormat.TSV,
            '.gb': DataFormat.GENBANK,
            '.gbk': DataFormat.GENBANK,
            '.json': DataFormat.JSON
        }
        
        detected = format_map.get(suffix)
        if detected:
            logger.info(f"检测到文件格式: {detected}")
            return detected
        
        # 尝试内容检测
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('>'):
                    return DataFormat.FASTA
                elif first_line.startswith('LOCUS'):
                    return DataFormat.GENBANK
                elif first_line.startswith('{') or first_line.startswith('['):
                    return DataFormat.JSON
                else:
                    return DataFormat.CSV
        except Exception as e:
            logger.warning(f"格式检测失败: {e}")
            return DataFormat.CSV
    
    def load_fasta(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """加载FASTA文件"""
        data = []
        try:
            for record in SeqIO.parse(file_path, 'fasta'):
                sequence = str(record.seq).upper()
                is_valid, error = self.validator.validate_sequence(sequence)
                
                item = {
                    'id': record.id,
                    'description': record.description,
                    'sequence': sequence,
                    'length': len(sequence),
                    'is_valid': is_valid,
                    'error': error
                }
                
                if is_valid:
                    item.update(self.validator.calculate_quality_metrics(sequence))
                
                data.append(item)
            
            logger.info(f"加载FASTA文件: {len(data)}条记录")
        except Exception as e:
            logger.error(f"FASTA文件加载失败: {e}")
            raise
        
        return data
    
    def load_csv(self, file_path: Union[str, Path], 
                 sequence_column: str = 'sequence',
                 label_column: Optional[str] = None) -> List[Dict[str, Any]]:
        """加载CSV文件"""
        data = []
        try:
            df = pd.read_csv(file_path)
            
            if sequence_column not in df.columns:
                raise ValueError(f"找不到序列列: {sequence_column}")
            
            for idx, row in df.iterrows():
                sequence = str(row[sequence_column]).upper().strip()
                is_valid, error = self.validator.validate_sequence(sequence)
                
                item = {
                    'id': f"seq_{idx}",
                    'sequence': sequence,
                    'length': len(sequence),
                    'is_valid': is_valid,
                    'error': error
                }
                
                # 添加标签
                if label_column and label_column in df.columns:
                    item['label'] = row[label_column]
                
                # 添加其他列
                for col in df.columns:
                    if col not in [sequence_column, label_column]:
                        item[col] = row[col]
                
                if is_valid:
                    item.update(self.validator.calculate_quality_metrics(sequence))
                
                data.append(item)
            
            logger.info(f"加载CSV文件: {len(data)}条记录")
        except Exception as e:
            logger.error(f"CSV文件加载失败: {e}")
            raise
        
        return data
    
    def load_genbank(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """加载GenBank文件"""
        data = []
        try:
            for record in SeqIO.parse(file_path, 'genbank'):
                sequence = str(record.seq).upper()
                is_valid, error = self.validator.validate_sequence(sequence)
                
                item = {
                    'id': record.id,
                    'description': record.description,
                    'sequence': sequence,
                    'length': len(sequence),
                    'organism': record.annotations.get('organism', ''),
                    'source': record.annotations.get('source', ''),
                    'is_valid': is_valid,
                    'error': error
                }
                
                # 提取特征信息
                features = []
                for feature in record.features:
                    features.append({
                        'type': feature.type,
                        'location': str(feature.location),
                        'qualifiers': dict(feature.qualifiers)
                    })
                item['features'] = features
                
                if is_valid:
                    item.update(self.validator.calculate_quality_metrics(sequence))
                
                data.append(item)
            
            logger.info(f"加载GenBank文件: {len(data)}条记录")
        except Exception as e:
            logger.error(f"GenBank文件加载失败: {e}")
            raise
        
        return data
    
    def load_json(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        data = []
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            if isinstance(json_data, list):
                raw_data = json_data
            elif isinstance(json_data, dict) and 'sequences' in json_data:
                raw_data = json_data['sequences']
            else:
                raw_data = [json_data]
            
            for idx, item in enumerate(raw_data):
                if isinstance(item, dict) and 'sequence' in item:
                    sequence = str(item['sequence']).upper().strip()
                    is_valid, error = self.validator.validate_sequence(sequence)
                    
                    processed_item = {
                        'id': item.get('id', f"seq_{idx}"),
                        'sequence': sequence,
                        'length': len(sequence),
                        'is_valid': is_valid,
                        'error': error
                    }
                    
                    # 复制其他字段
                    for key, value in item.items():
                        if key not in ['sequence', 'id']:
                            processed_item[key] = value
                    
                    if is_valid:
                        processed_item.update(self.validator.calculate_quality_metrics(sequence))
                    
                    data.append(processed_item)
            
            logger.info(f"加载JSON文件: {len(data)}条记录")
        except Exception as e:
            logger.error(f"JSON文件加载失败: {e}")
            raise
        
        return data
    
    def load_data(self, file_path: Union[str, Path], 
                  file_format: Optional[str] = None,
                  **kwargs) -> List[Dict[str, Any]]:
        """统一数据加载接口"""
        if file_format is None:
            file_format = self.detect_format(file_path)
        
        if file_format not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        logger.info(f"加载数据文件: {file_path}, 格式: {file_format}")
        
        if file_format == DataFormat.FASTA:
            return self.load_fasta(file_path)
        elif file_format == DataFormat.CSV:
            return self.load_csv(file_path, **kwargs)
        elif file_format == DataFormat.TSV:
            kwargs['sep'] = '\t'
            return self.load_csv(file_path, **kwargs)
        elif file_format == DataFormat.GENBANK:
            return self.load_genbank(file_path)
        elif file_format == DataFormat.JSON:
            return self.load_json(file_path)
        else:
            raise ValueError(f"未实现的格式加载器: {file_format}")

class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        self.validator = SequenceValidator()
    
    def check_dataset_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查数据集质量"""
        if not data:
            return {'error': '数据集为空'}
        
        total_count = len(data)
        valid_count = sum(1 for item in data if item.get('is_valid', False))
        invalid_count = total_count - valid_count
        
        # 统计序列长度
        lengths = [item['length'] for item in data if 'length' in item]
        length_stats = {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': np.mean(lengths) if lengths else 0,
            'std': np.std(lengths) if lengths else 0
        }
        
        # 统计GC含量
        gc_contents = [item['gc_content'] for item in data 
                      if 'gc_content' in item and item.get('is_valid', False)]
        gc_stats = {
            'min': min(gc_contents) if gc_contents else 0,
            'max': max(gc_contents) if gc_contents else 0,
            'mean': np.mean(gc_contents) if gc_contents else 0,
            'std': np.std(gc_contents) if gc_contents else 0
        }
        
        # 统计错误类型
        error_types = {}
        for item in data:
            if not item.get('is_valid', True) and item.get('error'):
                error = item['error']
                error_types[error] = error_types.get(error, 0) + 1
        
        quality_report = {
            'total_sequences': total_count,
            'valid_sequences': valid_count,
            'invalid_sequences': invalid_count,
            'validity_rate': valid_count / total_count if total_count > 0 else 0,
            'length_statistics': length_stats,
            'gc_content_statistics': gc_stats,
            'error_types': error_types,
            'recommendations': self._generate_recommendations(
                valid_count / total_count if total_count > 0 else 0,
                length_stats, gc_stats, error_types
            )
        }
        
        return quality_report
    
    def _generate_recommendations(self, validity_rate: float, 
                                length_stats: Dict, gc_stats: Dict,
                                error_types: Dict) -> List[str]:
        """生成数据质量改进建议"""
        recommendations = []
        
        if validity_rate < 0.9:
            recommendations.append(f"序列有效性较低({validity_rate:.2%})，建议进行数据清洗")
        
        if length_stats.get('std', 0) > length_stats.get('mean', 0) * 0.5:
            recommendations.append("序列长度差异较大，建议进行长度标准化")
        
        if gc_stats.get('std', 0) > 0.2:
            recommendations.append("GC含量分布较广，可能需要分层处理")
        
        if error_types:
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            recommendations.append(f"最常见错误类型：{most_common_error[0]} ({most_common_error[1]}次)")
        
        if not recommendations:
            recommendations.append("数据质量良好，可以直接使用")
        
        return recommendations
    
    def clean_dataset(self, data: List[Dict[str, Any]], 
                     min_length: int = 50, max_length: int = 2000,
                     min_gc: float = 0.2, max_gc: float = 0.8) -> List[Dict[str, Any]]:
        """清洗数据集"""
        cleaned_data = []
        removed_count = 0
        
        for item in data:
            # 基本有效性检查
            if not item.get('is_valid', False):
                removed_count += 1
                continue
            
            # 长度过滤
            length = item.get('length', 0)
            if length < min_length or length > max_length:
                removed_count += 1
                continue
            
            # GC含量过滤
            gc_content = item.get('gc_content', 0.5)
            if gc_content < min_gc or gc_content > max_gc:
                removed_count += 1
                continue
            
            cleaned_data.append(item)
        
        logger.info(f"数据清洗完成: 保留{len(cleaned_data)}条，移除{removed_count}条")
        return cleaned_data

class EnhancedDataset:
    """增强数据集类"""
    
    def __init__(self, max_length: Optional[int] = None, 
                 vocab_size: int = 4, enable_augmentation: bool = True):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.enable_augmentation = enable_augmentation
        
        self.loader = MultiFormatDataLoader()
        self.quality_checker = DataQualityChecker()
        self.augmentor = DataAugmentor() if enable_augmentation else None
        
        # 核苷酸映射
        self.nucleotide_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self.idx_to_nucleotide = {v: k for k, v in self.nucleotide_to_idx.items()}
        
        # 数据存储
        self.raw_data = []
        self.processed_data = []
        self.labels = []
        self.metadata = {}
    
    def load_from_file(self, file_path: Union[str, Path], 
                       file_format: Optional[str] = None,
                       **kwargs) -> 'EnhancedDataset':
        """从文件加载数据"""
        self.raw_data = self.loader.load_data(file_path, file_format, **kwargs)
        logger.info(f"加载数据: {len(self.raw_data)}条记录")
        return self
    
    def add_sequences(self, sequences: List[str], 
                      labels: Optional[List] = None,
                      metadata: Optional[List[Dict]] = None) -> 'EnhancedDataset':
        """直接添加序列数据"""
        validator = SequenceValidator()
        
        for i, seq in enumerate(sequences):
            is_valid, error = validator.validate_sequence(seq)
            
            item = {
                'id': f"seq_{len(self.raw_data) + i}",
                'sequence': seq.upper(),
                'length': len(seq),
                'is_valid': is_valid,
                'error': error
            }
            
            if labels is not None and i < len(labels):
                item['label'] = labels[i]
            
            if metadata is not None and i < len(metadata):
                item.update(metadata[i])
            
            if is_valid:
                item.update(validator.calculate_quality_metrics(seq))
            
            self.raw_data.append(item)
        
        logger.info(f"添加序列: {len(sequences)}条")
        return self
    
    def quality_check(self) -> Dict[str, Any]:
        """执行质量检查"""
        report = self.quality_checker.check_dataset_quality(self.raw_data)
        self.metadata['quality_report'] = report
        
        logger.info(f"质量检查完成: {report['validity_rate']:.2%}有效序列")
        return report
    
    def clean_data(self, **kwargs) -> 'EnhancedDataset':
        """清洗数据"""
        if not self.raw_data:
            logger.warning("没有数据需要清洗")
            return self
        
        self.raw_data = self.quality_checker.clean_dataset(self.raw_data, **kwargs)
        return self
    
    def encode_sequences(self, sequences: List[str]) -> torch.Tensor:
        """编码序列为数值张量"""
        encoded = []
        
        for seq in sequences:
            # 截断或填充到固定长度
            if self.max_length:
                if len(seq) > self.max_length:
                    seq = seq[:self.max_length]
                else:
                    seq = seq + 'N' * (self.max_length - len(seq))
            
            # 转换为索引
            indices = []
            for nucleotide in seq.upper():
                idx = self.nucleotide_to_idx.get(nucleotide, 0)  # N默认映射到A
                indices.append(idx)
            
            encoded.append(indices)
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def decode_sequences(self, encoded: torch.Tensor) -> List[str]:
        """解码数值张量为序列"""
        decoded = []
        
        for seq_tensor in encoded:
            seq = ''
            for idx in seq_tensor:
                nucleotide = self.idx_to_nucleotide.get(idx.item(), 'N')
                seq += nucleotide
            decoded.append(seq)
        
        return decoded
    
    def prepare_training_data(self, test_size: float = 0.2, 
                            validation_size: float = 0.1,
                            apply_augmentation: bool = True) -> Dict[str, torch.Tensor]:
        """准备训练数据"""
        if not self.raw_data:
            raise ValueError("没有数据可用于训练")
        
        # 提取有效序列
        valid_data = [item for item in self.raw_data if item.get('is_valid', False)]
        if not valid_data:
            raise ValueError("没有有效序列")
        
        sequences = [item['sequence'] for item in valid_data]
        labels = [item.get('label', 0) for item in valid_data]
        
        # 数据增强
        if apply_augmentation and self.augmentor:
            sequences, labels = self.augmentor.augment_batch(sequences, labels)
            logger.info(f"数据增强后: {len(sequences)}条序列")
        
        # 编码序列
        encoded_sequences = self.encode_sequences(sequences)
        
        # 处理标签
        if all(isinstance(label, (int, float)) for label in labels):
            labels_tensor = torch.tensor(labels, dtype=torch.float)
        else:
            # 字符串标签需要编码
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)
            labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)
            self.metadata['label_encoder'] = label_encoder
        
        # 数据分割
        if validation_size > 0:
            # 三分割：训练、验证、测试
            X_temp, X_test, y_temp, y_test = train_test_split(
                encoded_sequences, labels_tensor, test_size=test_size, random_state=42
            )
            val_ratio = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=42
            )
            
            return {
                'train_sequences': X_train,
                'train_labels': y_train,
                'val_sequences': X_val,
                'val_labels': y_val,
                'test_sequences': X_test,
                'test_labels': y_test
            }
        else:
            # 二分割：训练、测试
            X_train, X_test, y_train, y_test = train_test_split(
                encoded_sequences, labels_tensor, test_size=test_size, random_state=42
            )
            
            return {
                'train_sequences': X_train,
                'train_labels': y_train,
                'test_sequences': X_test,
                'test_labels': y_test
            }
    
    def save_processed_data(self, file_path: Union[str, Path], 
                          format: str = 'json') -> None:
        """保存处理后的数据"""
        file_path = Path(file_path)
        
        if format == 'json':
            data_to_save = {
                'raw_data': self.raw_data,
                'metadata': self.metadata,
                'config': {
                    'max_length': self.max_length,
                    'vocab_size': self.vocab_size,
                    'enable_augmentation': self.enable_augmentation
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
        
        elif format == 'csv':
            df = pd.DataFrame(self.raw_data)
            df.to_csv(file_path, index=False)
        
        elif format == 'fasta':
            records = []
            for item in self.raw_data:
                if item.get('is_valid', False):
                    record = SeqRecord(
                        Seq(item['sequence']),
                        id=item['id'],
                        description=item.get('description', '')
                    )
                    records.append(record)
            
            SeqIO.write(records, file_path, 'fasta')
        
        logger.info(f"数据已保存到: {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self.raw_data:
            return {'error': '没有数据'}
        
        valid_data = [item for item in self.raw_data if item.get('is_valid', False)]
        
        stats = {
            'total_sequences': len(self.raw_data),
            'valid_sequences': len(valid_data),
            'sequence_length_range': {
                'min': min(item['length'] for item in valid_data) if valid_data else 0,
                'max': max(item['length'] for item in valid_data) if valid_data else 0
            },
            'gc_content_range': {
                'min': min(item.get('gc_content', 0) for item in valid_data) if valid_data else 0,
                'max': max(item.get('gc_content', 0) for item in valid_data) if valid_data else 0
            }
        }
        
        return stats

def create_enhanced_dataset(max_length: Optional[int] = None,
                          vocab_size: int = 4,
                          enable_augmentation: bool = True) -> EnhancedDataset:
    """创建增强数据集的工厂函数"""
    return EnhancedDataset(
        max_length=max_length,
        vocab_size=vocab_size,
        enable_augmentation=enable_augmentation
    )
