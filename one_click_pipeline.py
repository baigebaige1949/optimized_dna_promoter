# -*- coding: utf-8 -*-
"""一键式训练分析流程

集成从数据加载、训练、评估、结果分析的完整流程，支持自动参数调优和结果可视化
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 导入我们的模块
from models.multimodal_fusion import create_multimodal_fusion_model, MultiModalPredictor
from training.advanced_trainer import create_advanced_trainer, AdvancedTrainer
from evaluation.biological_metrics import evaluate_generated_sequences, BiologicalMetrics
from utils.logger import setup_logger
from config.model_config import ModelConfig
from data.data_loader import DNASequenceDataset

class DNAPromote​Dataset(Dataset):
    """DNA启动子数据集类"""
    
    def __init__(self, sequences: List[str], labels: Optional[List[float]] = None, 
                 tokenizer: Optional[Dict[str, int]] = None, max_length: int = 1000):
        
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
        # 默认核苷酸标记器
        if tokenizer is None:
            self.tokenizer = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}
        else:
            self.tokenizer = tokenizer
            
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 序列编码
        encoded_seq = self._encode_sequence(sequence)
        
        item = {
            'input_ids': torch.tensor(encoded_seq, dtype=torch.long),
            'attention_mask': torch.ones(len(encoded_seq), dtype=torch.long)
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item
        
    def _encode_sequence(self, sequence: str) -> List[int]:
        """编码DNA序列"""
        sequence = sequence.upper()
        
        # 截断或填充到指定长度
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence = sequence + 'N' * (self.max_length - len(sequence))
            
        # 转换为数字编码
        encoded = [self.tokenizer.get(nt, 0) for nt in sequence]
        return encoded

class OneClickPipeline:
    """一键式训练分析流程"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化一键式流程
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        
        # 加载配置
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    self.config = json.load(f)
        else:
            self.config = self._get_default_config()
            
        # 创建输出目录
        self.output_dir = Path(self.config.get('output_dir', 'pipeline_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name='OneClickPipeline',
            log_file=self.output_dir / 'pipeline.log',
            level=logging.INFO
        )
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化组件
        self.model = None
        self.trainer = None
        self.evaluator = BiologicalMetrics()
        
        # 记录实验历史
        self.experiment_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 数据配置
            'data': {
                'train_path': None,
                'val_path': None,
                'test_path': None,
                'max_length': 1000,
                'val_split': 0.2,
                'test_split': 0.1,
                'batch_size': 32,
                'num_workers': 4
            },
            
            # 模型配置
            'model': {
                'vocab_size': 5,
                'seq_len': 1000,
                'embed_dim': 512,
                'hidden_dim': 512,
                'output_dim': 256
            },
            
            # 训练配置
            'training': {
                'num_epochs': 100,
                'use_amp': True,
                'max_grad_norm': 1.0,
                'early_stopping_patience': 20,
                'save_interval': 10,
                'eval_interval': 5,
                'log_interval': 100,
                'distributed': False,
                
                'optimizer': {
                    'lr': 1e-4,
                    'weight_decay': 1e-2,
                    'betas': [0.9, 0.999]
                },
                
                'scheduler': {
                    'type': 'cosine',
                    'T_max': 100,
                    'eta_min': 1e-6
                }
            },
            
            # 评估配置
            'evaluation': {
                'metrics': ['mse', 'mae', 'r2'],
                'biological_evaluation': True,
                'generate_plots': True
            },
            
            # 超参数调优配置
            'hyperparameter_tuning': {
                'enabled': False,
                'n_trials': 50,
                'pruning': True,
                'study_name': 'dna_promoter_optimization'
            },
            
            # 输出配置
            'output_dir': 'pipeline_results',
            'save_model': True,
            'save_predictions': True,
            'generate_report': True
        }
        
    def load_data(self, train_data: Union[str, List[str], pd.DataFrame] = None,
                  train_labels: Optional[List[float]] = None,
                  val_data: Union[str, List[str], pd.DataFrame] = None,
                  val_labels: Optional[List[float]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        加载和准备数据
        
        Args:
            train_data: 训练数据（文件路径、序列列表或DataFrame）
            train_labels: 训练标签
            val_data: 验证数据
            val_labels: 验证标签
            
        Returns:
            train_loader, val_loader, test_loader
        """
        
        self.logger.info("开始加载数据...")
        
        # 解析数据
        if isinstance(train_data, str):
            # 从文件加载
            if train_data.endswith('.csv'):
                df = pd.read_csv(train_data)
                sequences = df['sequence'].tolist()
                labels = df.get('label', df.get('strength', None))
                if labels is not None:
                    labels = labels.tolist()
            elif train_data.endswith('.fasta') or train_data.endswith('.fa'):
                sequences = self._load_fasta(train_data)
                labels = train_labels
            else:
                raise ValueError(f"不支持的文件格式: {train_data}")
                
        elif isinstance(train_data, pd.DataFrame):
            sequences = train_data['sequence'].tolist()
            labels = train_data.get('label', train_data.get('strength', None))
            if labels is not None:
                labels = labels.tolist()
                
        elif isinstance(train_data, list):
            sequences = train_data
            labels = train_labels
            
        else:
            raise ValueError("train_data必须是文件路径、序列列表或DataFrame")
            
        # 数据分割
        if val_data is None:
            val_split = self.config['data']['val_split']
            test_split = self.config['data']['test_split']
            
            if labels is not None:
                train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
                    sequences, labels, test_size=val_split + test_split, random_state=42
                )
                val_seqs, test_seqs, val_labels, test_labels = train_test_split(
                    temp_seqs, temp_labels, test_size=test_split/(val_split + test_split), random_state=42
                )
            else:
                train_seqs, temp_seqs = train_test_split(
                    sequences, test_size=val_split + test_split, random_state=42
                )
                val_seqs, test_seqs = train_test_split(
                    temp_seqs, test_size=test_split/(val_split + test_split), random_state=42
                )
                train_labels = val_labels = test_labels = None
        else:
            train_seqs, train_labels = sequences, labels
            if isinstance(val_data, str) and val_data.endswith('.csv'):
                val_df = pd.read_csv(val_data)
                val_seqs = val_df['sequence'].tolist()
                val_labels = val_df.get('label', val_df.get('strength', None))
                if val_labels is not None:
                    val_labels = val_labels.tolist()
            else:
                val_seqs, val_labels = val_data, val_labels
                
            # 测试集暂时设为验证集
            test_seqs, test_labels = val_seqs, val_labels
            
        self.logger.info(f"数据分割完成: 训练集 {len(train_seqs)}, 验证集 {len(val_seqs)}, 测试集 {len(test_seqs)}")
        
        # 创建数据集
        max_length = self.config['data']['max_length']
        
        train_dataset = DNAPromote​Dataset(train_seqs, train_labels, max_length=max_length)
        val_dataset = DNAPromote​Dataset(val_seqs, val_labels, max_length=max_length)
        test_dataset = DNAPromote​Dataset(test_seqs, test_labels, max_length=max_length)
        
        # 创建数据加载器
        batch_size = self.config['data']['batch_size']
        num_workers = self.config['data']['num_workers']
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # 保存数据统计信息
        self._save_data_statistics(train_seqs, val_seqs, test_seqs)
        
        return train_loader, val_loader, test_loader
        
    def _load_fasta(self, fasta_path: str) -> List[str]:
        """加载FASTA文件"""
        sequences = []
        try:
            from Bio import SeqIO
            for record in SeqIO.parse(fasta_path, "fasta"):
                sequences.append(str(record.seq))
        except ImportError:
            # 如果没有BioPython，使用简单的解析器
            with open(fasta_path, 'r') as f:
                sequence = ""
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if sequence:
                            sequences.append(sequence)
                            sequence = ""
                    else:
                        sequence += line
                if sequence:
                    sequences.append(sequence)
                    
        return sequences
        
    def _save_data_statistics(self, train_seqs: List[str], val_seqs: List[str], test_seqs: List[str]):
        """保存数据统计信息"""
        
        stats = {
            'train': {
                'num_sequences': len(train_seqs),
                'avg_length': np.mean([len(seq) for seq in train_seqs]),
                'min_length': min([len(seq) for seq in train_seqs]),
                'max_length': max([len(seq) for seq in train_seqs])
            },
            'validation': {
                'num_sequences': len(val_seqs),
                'avg_length': np.mean([len(seq) for seq in val_seqs]),
                'min_length': min([len(seq) for seq in val_seqs]),
                'max_length': max([len(seq) for seq in val_seqs])
            },
            'test': {
                'num_sequences': len(test_seqs),
                'avg_length': np.mean([len(seq) for seq in test_seqs]),
                'min_length': min([len(seq) for seq in test_seqs]),
                'max_length': max([len(seq) for seq in test_seqs])
            }
        }
        
        # 保存统计信息
        with open(self.output_dir / 'data_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        self.logger.info("数据统计信息已保存")
        
    def create_model(self, trial: Optional[optuna.Trial] = None) -> MultiModalPredictor:
        """
        创建模型
        
        Args:
            trial: Optuna试验对象，用于超参数调优
            
        Returns:
            创建的模型
        """
        
        if trial is not None:
            # 超参数调优模式
            config = {
                'vocab_size': self.config['model']['vocab_size'],
                'seq_len': self.config['model']['seq_len'],
                'embed_dim': trial.suggest_categorical('embed_dim', [256, 512, 768]),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 768]),
                'output_dim': trial.suggest_categorical('output_dim', [128, 256, 512])
            }
        else:
            # 使用配置文件中的参数
            config = self.config['model']
            
        self.model = create_multimodal_fusion_model(config)
        self.model = self.model.to(self.device)
        
        self.logger.info(f"模型创建完成，参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
        
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   trial: Optional[optuna.Trial] = None) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            trial: Optuna试验对象
            
        Returns:
            训练统计信息
        """
        
        if self.model is None:
            raise ValueError("请先创建模型")
            
        # 准备训练配置
        training_config = self.config['training'].copy()
        
        if trial is not None:
            # 超参数调优模式
            training_config['optimizer']['lr'] = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            training_config['optimizer']['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)
            
        training_config['output_dir'] = str(self.output_dir / 'checkpoints')
        
        # 创建训练器
        self.trainer = create_advanced_trainer(
            self.model, 
            training_config, 
            self.device,
            self.logger
        )
        
        # 定义损失函数
        def criterion(outputs, batch):
            predictions = outputs['predictions'].squeeze()
            targets = batch['labels']
            return nn.MSELoss()(predictions, targets)
            
        # 定义评估指标
        def mae_metric(outputs, batch):
            predictions = outputs['predictions'].squeeze()
            targets = batch['labels']
            return torch.mean(torch.abs(predictions - targets)).item()
            
        def r2_metric(outputs, batch):
            predictions = outputs['predictions'].squeeze()
            targets = batch['labels']
            
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            return r2.item()
            
        metrics = [mae_metric, r2_metric]
        
        self.logger.info("开始训练...")
        
        # 训练模型
        training_stats = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            metrics=metrics
        )
        
        # 保存训练统计信息
        self.trainer.save_training_stats('training_stats.json')
        
        self.logger.info("训练完成!")
        
        return training_stats
        
    def evaluate_model(self, test_loader: DataLoader, generate_sequences: bool = True) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            generate_sequences: 是否生成序列进行生物学评估
            
        Returns:
            评估结果
        """
        
        if self.model is None or self.trainer is None:
            raise ValueError("请先训练模型")
            
        self.logger.info("开始模型评估...")
        
        # 基本评估指标
        def criterion(outputs, batch):
            predictions = outputs['predictions'].squeeze()
            targets = batch['labels']
            return nn.MSELoss()(predictions, targets)
            
        def mae_metric(outputs, batch):
            predictions = outputs['predictions'].squeeze()
            targets = batch['labels']
            return torch.mean(torch.abs(predictions - targets)).item()
            
        def r2_metric(outputs, batch):
            predictions = outputs['predictions'].squeeze()
            targets = batch['labels']
            
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            return r2.item()
            
        metrics = [mae_metric, r2_metric]
        
        # 运行评估
        eval_results = self.trainer.evaluate(test_loader, criterion, metrics)
        
        # 获取预测结果
        predictions, targets, sequences = self._get_predictions(test_loader)
        
        # 保存预测结果
        if self.config['output'].get('save_predictions', True):
            self._save_predictions(predictions, targets, sequences)
            
        results = {
            'basic_metrics': eval_results,
            'predictions': {
                'mse': float(np.mean((predictions - targets) ** 2)),
                'mae': float(np.mean(np.abs(predictions - targets))),
                'r2': float(1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
            }
        }
        
        # 生物学评估
        if self.config['evaluation'].get('biological_evaluation', True) and generate_sequences:
            self.logger.info("开始生物学评估...")
            
            # 这里应该有序列生成逻辑，目前简化处理
            # 实际应用中需要实现序列生成功能
            real_sequences = sequences[:100]  # 使用真实序列作为参考
            generated_sequences = self._generate_dummy_sequences(len(real_sequences))
            
            bio_results = evaluate_generated_sequences(
                real_sequences,
                generated_sequences,
                output_dir=str(self.output_dir / 'biological_evaluation')
            )
            
            results['biological_metrics'] = bio_results
            
        self.logger.info("评估完成!")
        
        return results
        
    def _get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """获取模型预测结果"""
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_sequences = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 准备输入数据
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['labels']
                
                # 模型预测
                outputs = self.model(input_ids)
                predictions = outputs['predictions'].cpu().squeeze()
                
                all_predictions.extend(predictions.numpy())
                all_targets.extend(targets.numpy())
                
                # 解码序列（简化版本）
                sequences = self._decode_sequences(input_ids.cpu())
                all_sequences.extend(sequences)
                
        return np.array(all_predictions), np.array(all_targets), all_sequences
        
    def _decode_sequences(self, encoded_sequences: torch.Tensor) -> List[str]:
        """解码序列"""
        
        idx_to_nt = {1: 'A', 2: 'T', 3: 'G', 4: 'C', 0: 'N'}
        sequences = []
        
        for seq in encoded_sequences:
            decoded = ''.join([idx_to_nt.get(idx.item(), 'N') for idx in seq])
            # 移除填充的N
            decoded = decoded.rstrip('N')
            sequences.append(decoded)
            
        return sequences
        
    def _generate_dummy_sequences(self, num_sequences: int) -> List[str]:
        """生成虚拟序列用于生物学评估测试"""
        sequences = []
        for _ in range(num_sequences):
            length = np.random.randint(100, 1000)
            seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], length))
            sequences.append(seq)
        return sequences
        
    def _save_predictions(self, predictions: np.ndarray, targets: np.ndarray, sequences: List[str]):
        """保存预测结果"""
        
        df = pd.DataFrame({
            'sequence': sequences,
            'true_strength': targets,
            'predicted_strength': predictions,
            'absolute_error': np.abs(predictions - targets),
            'relative_error': np.abs(predictions - targets) / (targets + 1e-8)
        })
        
        df.to_csv(self.output_dir / 'predictions.csv', index=False)
        self.logger.info("预测结果已保存")
        
    def hyperparameter_tuning(self, train_loader: DataLoader, val_loader: DataLoader) -> optuna.Study:
        """
        超参数调优
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            Optuna研究对象
        """
        
        if not self.config['hyperparameter_tuning']['enabled']:
            self.logger.info("超参数调优未启用")
            return None
            
        self.logger.info("开始超参数调优...")
        
        def objective(trial):
            try:
                # 创建模型
                model = self.create_model(trial)
                
                # 训练模型（减少epoch数以提高调优速度）
                original_epochs = self.config['training']['num_epochs']
                self.config['training']['num_epochs'] = min(20, original_epochs)
                
                training_stats = self.train_model(train_loader, val_loader, trial)
                
                # 恢复原始epoch数
                self.config['training']['num_epochs'] = original_epochs
                
                # 返回验证损失作为优化目标
                val_losses = training_stats.get('val_loss', [float('inf')])
                return min(val_losses) if val_losses else float('inf')
                
            except Exception as e:
                self.logger.error(f"Trial失败: {e}")
                return float('inf')
                
        # 创建研究
        study_name = self.config['hyperparameter_tuning']['study_name']
        n_trials = self.config['hyperparameter_tuning']['n_trials']
        
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner() if self.config['hyperparameter_tuning']['pruning'] else None
        )
        
        # 运行优化
        study.optimize(objective, n_trials=n_trials)
        
        # 保存优化结果
        self._save_optimization_results(study)
        
        self.logger.info("超参数调优完成!")
        self.logger.info(f"最佳参数: {study.best_params}")
        self.logger.info(f"最佳分数: {study.best_value}")
        
        return study
        
    def _save_optimization_results(self, study: optuna.Study):
        """保存超参数优化结果"""
        
        # 保存最佳参数
        with open(self.output_dir / 'best_hyperparameters.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
            
        # 保存优化历史
        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.output_dir / 'optimization_trials.csv', index=False)
        
        # 创建优化可视化图
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 优化历史图
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=[trial.value for trial in study.trials if trial.value is not None],
                mode='lines+markers',
                name='目标值'
            ))
            fig.update_layout(
                title='超参数优化历史',
                xaxis_title='试验次数',
                yaxis_title='验证损失'
            )
            fig.write_html(self.output_dir / 'optimization_history.html')
            
            self.logger.info("优化可视化图已保存")
            
        except ImportError:
            self.logger.warning("plotly未安装，跳过可视化图生成")
            
    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        生成综合分析报告
        
        Args:
            evaluation_results: 评估结果
            
        Returns:
            报告文件路径
        """
        
        report_path = self.output_dir / 'comprehensive_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# DNA启动子强度预测模型 - 综合分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 模型配置
            f.write("## 模型配置\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config['model'], indent=2, ensure_ascii=False))
            f.write("\n```\n\n")
            
            # 训练配置
            f.write("## 训练配置\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config['training'], indent=2, ensure_ascii=False))
            f.write("\n```\n\n")
            
            # 评估结果
            f.write("## 评估结果\n\n")
            
            # 基础指标
            if 'basic_metrics' in evaluation_results:
                f.write("### 基础评估指标\n\n")
                for key, value in evaluation_results['basic_metrics'].items():
                    f.write(f"- **{key}**: {value:.6f}\n")
                f.write("\n")
                
            # 预测指标
            if 'predictions' in evaluation_results:
                f.write("### 预测性能指标\n\n")
                pred_metrics = evaluation_results['predictions']
                f.write(f"- **均方误差 (MSE)**: {pred_metrics['mse']:.6f}\n")
                f.write(f"- **平均绝对误差 (MAE)**: {pred_metrics['mae']:.6f}\n")
                f.write(f"- **R² 分数**: {pred_metrics['r2']:.6f}\n\n")
                
            # 生物学评估指标
            if 'biological_metrics' in evaluation_results:
                f.write("### 生物学评估指标\n\n")
                bio_metrics = evaluation_results['biological_metrics']
                
                if 'js_divergence_1mer' in bio_metrics:
                    f.write(f"- **JS散度 (1-mer)**: {bio_metrics['js_divergence_1mer']:.6f}\n")
                if 'js_divergence_2mer' in bio_metrics:
                    f.write(f"- **JS散度 (2-mer)**: {bio_metrics['js_divergence_2mer']:.6f}\n")
                if 'js_divergence_3mer' in bio_metrics:
                    f.write(f"- **JS散度 (3-mer)**: {bio_metrics['js_divergence_3mer']:.6f}\n")
                if 's_fid' in bio_metrics:
                    f.write(f"- **序列FID (S-FID)**: {bio_metrics['s_fid']:.6f}\n")
                    
                if 'gc_content' in bio_metrics:
                    gc_info = bio_metrics['gc_content']
                    f.write(f"- **GC含量差异**: {gc_info['difference']:.6f}\n")
                    
                f.write("\n")
                
            # 总结和建议
            f.write("## 总结和建议\n\n")
            
            # 根据结果生成建议
            if 'predictions' in evaluation_results:
                r2 = evaluation_results['predictions']['r2']
                if r2 > 0.8:
                    f.write("✅ **模型性能优秀**: R²分数超过0.8，模型具有很好的预测能力。\n\n")
                elif r2 > 0.6:
                    f.write("⚠️ **模型性能良好**: R²分数在0.6-0.8之间，模型有一定的预测能力，但仍有改进空间。\n\n")
                else:
                    f.write("❌ **模型性能需要改进**: R²分数低于0.6，建议调整模型架构或超参数。\n\n")
                    
            f.write("### 改进建议\n\n")
            f.write("1. **数据增强**: 考虑使用数据增强技术增加训练数据的多样性\n")
            f.write("2. **特征工程**: 引入更多生物学相关的特征\n")
            f.write("3. **模型集成**: 考虑使用模型集成方法提高预测性能\n")
            f.write("4. **超参数调优**: 进一步优化模型超参数\n")
            f.write("5. **正则化**: 如果出现过拟合，增加正则化技术\n\n")
            
            # 附录
            f.write("## 附录\n\n")
            f.write("### 相关文件\n\n")
            f.write("- 训练统计: `training_stats.json`\n")
            f.write("- 预测结果: `predictions.csv`\n")
            f.write("- 数据统计: `data_statistics.json`\n")
            f.write("- 最佳模型: `checkpoints/best_model.pt`\n\n")
            
        self.logger.info(f"综合分析报告已生成: {report_path}")
        return str(report_path)
        
    def run_complete_pipeline(self, 
                            train_data: Union[str, List[str], pd.DataFrame],
                            train_labels: Optional[List[float]] = None,
                            val_data: Optional[Union[str, List[str], pd.DataFrame]] = None,
                            val_labels: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        运行完整的一键式流程
        
        Args:
            train_data: 训练数据
            train_labels: 训练标签
            val_data: 验证数据
            val_labels: 验证标签
            
        Returns:
            完整的流程结果
        """
        
        start_time = datetime.now()
        self.logger.info("开始一键式流程...")
        
        results = {}
        
        try:
            # 1. 数据加载
            self.logger.info("步骤1: 数据加载")
            train_loader, val_loader, test_loader = self.load_data(
                train_data, train_labels, val_data, val_labels
            )
            
            # 2. 超参数调优（如果启用）
            if self.config['hyperparameter_tuning']['enabled']:
                self.logger.info("步骤2: 超参数调优")
                study = self.hyperparameter_tuning(train_loader, val_loader)
                results['optimization_study'] = study
                
                # 使用最佳参数更新配置
                if study and study.best_params:
                    for key, value in study.best_params.items():
                        if key in self.config['model']:
                            self.config['model'][key] = value
                        elif key in self.config['training']['optimizer']:
                            self.config['training']['optimizer'][key] = value
                            
            # 3. 模型创建
            self.logger.info("步骤3: 模型创建")
            model = self.create_model()
            results['model'] = model
            
            # 4. 模型训练
            self.logger.info("步骤4: 模型训练")
            training_stats = self.train_model(train_loader, val_loader)
            results['training_stats'] = training_stats
            
            # 5. 模型评估
            self.logger.info("步骤5: 模型评估")
            evaluation_results = self.evaluate_model(test_loader)
            results['evaluation_results'] = evaluation_results
            
            # 6. 生成报告
            if self.config.get('generate_report', True):
                self.logger.info("步骤6: 生成综合报告")
                report_path = self.generate_report(evaluation_results)
                results['report_path'] = report_path
                
            # 7. 保存最终配置
            final_config_path = self.output_dir / 'final_config.yaml'
            with open(final_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                
            end_time = datetime.now()
            duration = end_time - start_time
            
            results['pipeline_info'] = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'output_directory': str(self.output_dir)
            }
            
            self.logger.info(f"一键式流程完成! 总耗时: {duration}")
            self.logger.info(f"结果保存在: {self.output_dir}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"流程执行失败: {e}")
            raise
            
def main():
    """主函数 - 演示如何使用一键式流程"""
    
    # 创建示例数据
    np.random.seed(42)
    
    # 生成模拟DNA序列和标签
    sequences = []
    labels = []
    
    for i in range(1000):
        # 生成随机DNA序列
        length = np.random.randint(100, 500)
        seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], length))
        sequences.append(seq)
        
        # 生成模拟的启动子强度标签（基于序列特征）
        gc_content = (seq.count('G') + seq.count('C')) / len(seq)
        # 简单的模拟规则：GC含量适中的序列强度较高
        strength = np.random.normal(0.5, 0.2) + 0.3 * (1 - abs(gc_content - 0.5) * 2)
        strength = np.clip(strength, 0, 1)  # 限制在[0,1]范围内
        labels.append(strength)
        
    print("生成示例数据完成")
    print(f"序列数量: {len(sequences)}")
    print(f"平均序列长度: {np.mean([len(seq) for seq in sequences]):.1f}")
    print(f"平均强度标签: {np.mean(labels):.3f}")
    
    # 创建配置
    config = {
        'data': {
            'max_length': 1000,
            'batch_size': 16,  # 减小批次大小以适应演示
            'num_workers': 0   # Windows兼容性
        },
        'model': {
            'embed_dim': 256,  # 减小模型大小
            'hidden_dim': 256,
            'output_dim': 128
        },
        'training': {
            'num_epochs': 10,  # 减少epoch数以加快演示
            'early_stopping_patience': 5,
            'optimizer': {
                'lr': 1e-3
            }
        },
        'hyperparameter_tuning': {
            'enabled': False  # 演示时关闭超参数调优
        },
        'output_dir': 'demo_pipeline_results'
    }
    
    # 保存配置文件
    with open('demo_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
    # 创建一键式流程
    pipeline = OneClickPipeline('demo_config.yaml')
    
    # 运行完整流程
    results = pipeline.run_complete_pipeline(sequences, labels)
    
    print("\n一键式流程完成!")
    print(f"结果保存在: {results['pipeline_info']['output_directory']}")
    print(f"总耗时: {results['pipeline_info']['duration_seconds']:.1f}秒")
    
    if 'evaluation_results' in results:
        eval_results = results['evaluation_results']
        if 'predictions' in eval_results:
            pred_metrics = eval_results['predictions']
            print(f"\n模型性能:")
            print(f"  MSE: {pred_metrics['mse']:.6f}")
            print(f"  MAE: {pred_metrics['mae']:.6f}")
            print(f"  R²: {pred_metrics['r2']:.6f}")

if __name__ == "__main__":
    main()
