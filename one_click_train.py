#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键式训练脚本

DNA启动子优化系统的完整训练和生成脚本
支持命令行参数配置和一键式运行

使用示例：
    python one_click_train.py --quick-demo
    python one_click_train.py --data-path data/promoters.csv --epochs 100
    python one_click_train.py --conditions temperature=37,ph=7.4,cell_type=ecoli
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入我们的模块
try:
    from models.multimodal_fusion import create_multimodal_fusion_model
    from training.advanced_trainer import create_advanced_trainer
    from evaluation.biological_metrics import BiologicalMetrics
    from conditions import create_condition_system
    from generation import create_generation_pipeline
    from data.enhanced_dataset import create_enhanced_dataset
    from utils.logger import setup_logger
    from utils.device_manager import DeviceManager
    from config.model_config import ModelConfig
    from config.training_config import TrainingConfig
except ImportError as e:
    print(f"模块导入错误: {e}")
    print("请确保在optimized_dna_promoter目录下运行此脚本")
    sys.exit(1)


class OneClickTrainer:
    """一键式训练器"""
    
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(
            name='OneClickTrainer',
            log_file=self.output_dir / 'training.log',
            level=logging.INFO
        )
        
        self.logger.info(f"一键式训练开始 - {self.start_time}")
        self.logger.info(f"输出目录: {self.output_dir}")
        
        # 设备管理
        self.device_manager = DeviceManager()
        self.device = self.device_manager.get_optimal_device()
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化组件
        self.model = None
        self.trainer = None
        self.dataset = None
        self.condition_controller = None
        self.condition_filler = None
        self.generation_pipeline = None
        self.evaluator = None
        
        # 训练历史
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'metrics': {},
            'best_epoch': 0,
            'best_score': 0.0
        }
        
    def setup_components(self):
        """初始化所有组件"""
        self.logger.info("初始化系统组件...")
        
        # 1. 创建条件控制系统
        self.logger.info("创建条件控制系统...")
        self.condition_controller, self.condition_filler = create_condition_system()
        
        # 2. 创建数据处理系统
        self.logger.info("创建数据处理系统...")
        self.dataset = create_enhanced_dataset(
            max_length=self.args.max_length,
            vocab_size=5,  # ATCGN
            enable_augmentation=self.args.data_augmentation
        )
        
        # 3. 创建模型
        self.logger.info("创建模型...")
        model_config = {
            'vocab_size': 5,
            'seq_len': self.args.max_length,
            'embed_dim': self.args.embed_dim,
            'hidden_dim': self.args.hidden_dim,
            'output_dim': 1,
            'condition_dim': self.condition_controller.get_condition_dim()
        }
        
        self.model = create_multimodal_fusion_model(**model_config)
        self.model = self.model.to(self.device)
        
        # 4. 创建训练器
        self.logger.info("创建训练器...")
        training_config = {
            'learning_rate': self.args.lr,
            'batch_size': self.args.batch_size,
            'num_epochs': self.args.epochs,
            'early_stopping_patience': self.args.patience,
            'use_amp': self.args.mixed_precision,
            'gradient_accumulation_steps': self.args.grad_accum_steps,
            'max_grad_norm': self.args.max_grad_norm,
            'save_interval': 10,
            'eval_interval': 5,
            'device': self.device
        }
        
        if self.args.distributed:
            training_config['distributed'] = True
            training_config['world_size'] = torch.cuda.device_count()
            
        self.trainer = create_advanced_trainer(
            model=self.model,
            config=training_config
        )
        
        # 5. 创建生成流水线
        self.logger.info("创建生成流水线...")
        generation_config = {
            'noise_scheduler': 'cosine',
            'sampler': 'dpm_solver_plus',
            'post_process': True,
            'biological_constraints': True
        }
        
        self.generation_pipeline = create_generation_pipeline(generation_config)
        
        # 6. 创建评估器
        self.logger.info("创建评估器...")
        self.evaluator = BiologicalMetrics()
        
        self.logger.info("所有组件初始化完成")
        
    def load_data(self):
        """加载和准备数据"""
        self.logger.info("开始数据加载和预处理...")
        
        if self.args.quick_demo:
            # 快速演示：生成示例数据
            self.logger.info("生成演示数据...")
            demo_sequences = self.generate_demo_sequences(1000)
            demo_labels = np.random.uniform(0.1, 1.0, len(demo_sequences))
            
            # 创建DataFrame
            demo_df = pd.DataFrame({
                'sequence': demo_sequences,
                'strength': demo_labels
            })
            
            # 保存演示数据
            demo_path = self.output_dir / 'demo_data.csv'
            demo_df.to_csv(demo_path, index=False)
            self.logger.info(f"演示数据保存至: {demo_path}")
            
            # 加载数据
            self.dataset.load_from_dataframe(demo_df)
            
        elif self.args.data_path:
            # 从文件加载数据
            self.logger.info(f"从文件加载数据: {self.args.data_path}")
            self.dataset.load_from_file(self.args.data_path)
            
        else:
            raise ValueError("必须指定数据路径或使用--quick-demo模式")
            
        # 数据质量检查
        self.logger.info("执行数据质量检查...")
        quality_report = self.dataset.quality_check()
        self.logger.info(f"数据质量报告: {quality_report}")
        
        # 数据清洗
        if not self.args.skip_data_clean:
            self.logger.info("执行数据清洗...")
            self.dataset.clean_data(
                min_length=50,
                max_length=self.args.max_length,
                min_gc=0.2,
                max_gc=0.8
            )
            
        # 准备训练数据
        self.logger.info("准备训练数据...")
        data_splits = self.dataset.prepare_training_data(
            test_size=0.2,
            validation_size=0.1,
            apply_augmentation=self.args.data_augmentation
        )
        
        self.train_dataset = data_splits['train']
        self.val_dataset = data_splits['val']
        self.test_dataset = data_splits['test']
        
        self.logger.info(f"数据分割完成:")
        self.logger.info(f"  训练集: {len(self.train_dataset)} 样本")
        self.logger.info(f"  验证集: {len(self.val_dataset)} 样本")
        self.logger.info(f"  测试集: {len(self.test_dataset)} 样本")
        
    def generate_demo_sequences(self, n_sequences: int) -> List[str]:
        """生成演示用的DNA序列"""
        bases = ['A', 'T', 'C', 'G']
        sequences = []
        
        for _ in range(n_sequences):
            # 生成长度在100-300之间的序列
            length = np.random.randint(100, min(300, self.args.max_length))
            sequence = ''.join(np.random.choice(bases, length))
            
            # 添加一些常见的启动子motif
            if np.random.random() < 0.3:  # 30%概率添加TATA box
                tata_pos = np.random.randint(20, length - 10)
                sequence = sequence[:tata_pos] + 'TATAAA' + sequence[tata_pos + 6:]
                
            if np.random.random() < 0.2:  # 20%概率添加CAAT box
                caat_pos = np.random.randint(10, length - 10)
                sequence = sequence[:caat_pos] + 'CCAAT' + sequence[caat_pos + 5:]
                
            sequences.append(sequence)
            
        return sequences
        
    def setup_conditions(self):
        """设置训练条件"""
        self.logger.info("设置训练条件...")
        
        # 解析命令行条件参数
        conditions = {}
        if self.args.conditions:
            for condition_str in self.args.conditions.split(','):
                key, value = condition_str.strip().split('=')
                # 尝试转换为数字
                try:
                    value = float(value)
                except ValueError:
                    pass  # 保持字符串格式
                conditions[key] = value
                
        # 如果没有指定条件，使用默认条件
        if not conditions:
            conditions = {
                'temperature': 37.0,
                'ph': 7.4,
                'cell_type': 'E.coli',
                'growth_phase': 'log'
            }
            
        self.logger.info(f"基础条件: {conditions}")
        
        # 智能填充缺失条件
        self.logger.info("执行智能条件填充...")
        self.target_conditions = self.condition_filler.intelligent_fill(
            conditions,
            biological_context='prokaryotic',
            target_pathways=['glycolysis']
        )
        
        self.logger.info(f"完整条件向量: {self.target_conditions.conditions}")
        
    def train_model(self):
        """训练模型"""
        self.logger.info("开始模型训练...")
        
        # 设置训练回调
        def training_callback(epoch, metrics):
            self.training_history['train_losses'].append(metrics.get('train_loss', 0))
            self.training_history['val_losses'].append(metrics.get('val_loss', 0))
            
            if metrics.get('val_score', 0) > self.training_history['best_score']:
                self.training_history['best_score'] = metrics['val_score']
                self.training_history['best_epoch'] = epoch
                
            # 每10轮保存一次训练历史
            if epoch % 10 == 0:
                self.save_training_history()
                
        # 开始训练
        training_results = self.trainer.train(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            callback=training_callback
        )
        
        self.logger.info("训练完成")
        self.logger.info(f"最佳验证分数: {self.training_history['best_score']:.4f}")
        self.logger.info(f"最佳轮次: {self.training_history['best_epoch']}")
        
        # 保存最终模型
        model_path = self.output_dir / 'best_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_config': self.trainer.config,
            'training_history': self.training_history,
            'conditions': self.target_conditions.conditions
        }, model_path)
        
        self.logger.info(f"模型保存至: {model_path}")
        
        return training_results
        
    def generate_sequences(self, n_sequences: int = 100):
        """生成DNA序列"""
        self.logger.info(f"开始生成 {n_sequences} 个DNA序列...")
        
        # 生成序列
        generated_sequences = self.generation_pipeline.generate(
            model=self.model,
            conditions=self.target_conditions.to_tensor().to(self.device),
            batch_size=min(32, n_sequences),
            seq_length=self.args.max_length,
            num_steps=50
        )
        
        # 限制数量
        generated_sequences = generated_sequences[:n_sequences]
        
        self.logger.info(f"成功生成 {len(generated_sequences)} 个序列")
        
        # 保存序列
        fasta_path = self.output_dir / 'generated_sequences.fasta'
        self.save_sequences_fasta(generated_sequences, fasta_path)
        
        csv_path = self.output_dir / 'generated_sequences.csv'
        self.save_sequences_csv(generated_sequences, csv_path)
        
        return generated_sequences
        
    def evaluate_sequences(self, sequences: List[str]):
        """评估生成的序列"""
        self.logger.info("开始序列评估...")
        
        # 生物学指标评估
        eval_results = self.evaluator.evaluate_batch(
            sequences=sequences,
            reference_conditions=self.target_conditions.conditions,
            metrics=['gc_content', 'complexity', 'motif_analysis', 'secondary_structure']
        )
        
        self.logger.info("评估结果:")
        for metric, value in eval_results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
            
        # 条件符合度检查
        compliance_report = self.evaluator.check_condition_compliance(
            sequences=sequences,
            target_conditions=self.target_conditions.conditions
        )
        
        self.logger.info("条件符合度:")
        for condition, compliance in compliance_report.items():
            self.logger.info(f"  {condition}: {compliance*100:.1f}%")
            
        # 保存评估报告
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'n_sequences': len(sequences),
            'biological_metrics': eval_results,
            'condition_compliance': compliance_report,
            'target_conditions': self.target_conditions.conditions
        }
        
        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"评估报告保存至: {report_path}")
        
        return evaluation_report
        
    def create_visualizations(self):
        """创建可视化图表"""
        self.logger.info("创建可视化图表...")
        
        # 1. 训练损失曲线
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_losses'], label='训练损失')
        plt.plot(self.training_history['val_losses'], label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True)
        
        # 2. 条件分布图
        plt.subplot(1, 2, 2)
        conditions = self.target_conditions.conditions
        numeric_conditions = {k: v for k, v in conditions.items() 
                            if isinstance(v, (int, float))}
        
        if numeric_conditions:
            keys = list(numeric_conditions.keys())
            values = list(numeric_conditions.values())
            plt.bar(keys, values)
            plt.xlabel('条件类型')
            plt.ylabel('条件值')
            plt.title('目标条件分布')
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plot_path = self.output_dir / 'training_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"可视化图表保存至: {plot_path}")
        
    def save_sequences_fasta(self, sequences: List[str], path: Path):
        """保存序列为FASTA格式"""
        with open(path, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">generated_sequence_{i+1}\n")
                f.write(f"{seq}\n")
                
    def save_sequences_csv(self, sequences: List[str], path: Path):
        """保存序列为CSV格式"""
        df = pd.DataFrame({
            'sequence_id': [f"seq_{i+1}" for i in range(len(sequences))],
            'sequence': sequences,
            'length': [len(seq) for seq in sequences],
            'gc_content': [self.calculate_gc_content(seq) for seq in sequences]
        })
        df.to_csv(path, index=False)
        
    def calculate_gc_content(self, sequence: str) -> float:
        """计算GC含量"""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0.0
        
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
    def generate_final_report(self):
        """生成最终报告"""
        end_time = datetime.now()
        total_time = end_time - self.start_time
        
        report = {
            'execution_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time': str(total_time),
                'success': True
            },
            'configuration': {
                'model_config': {
                    'embed_dim': self.args.embed_dim,
                    'hidden_dim': self.args.hidden_dim,
                    'max_length': self.args.max_length
                },
                'training_config': {
                    'epochs': self.args.epochs,
                    'batch_size': self.args.batch_size,
                    'learning_rate': self.args.lr,
                    'mixed_precision': self.args.mixed_precision
                },
                'target_conditions': self.target_conditions.conditions
            },
            'results': {
                'best_epoch': self.training_history['best_epoch'],
                'best_score': self.training_history['best_score'],
                'final_train_loss': self.training_history['train_losses'][-1] 
                                  if self.training_history['train_losses'] else None,
                'final_val_loss': self.training_history['val_losses'][-1] 
                                if self.training_history['val_losses'] else None
            },
            'generated_files': {
                'model': 'best_model.pth',
                'sequences_fasta': 'generated_sequences.fasta',
                'sequences_csv': 'generated_sequences.csv',
                'evaluation': 'evaluation_report.json',
                'visualizations': 'training_analysis.png',
                'logs': 'training.log'
            }
        }
        
        report_path = self.output_dir / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"最终报告保存至: {report_path}")
        self.logger.info(f"总运行时间: {total_time}")
        
        # 打印总结
        print("\n" + "="*60)
        print("DNA启动子优化系统 - 一键式训练完成")
        print("="*60)
        print(f"输出目录: {self.output_dir}")
        print(f"运行时间: {total_time}")
        print(f"最佳轮次: {self.training_history['best_epoch']}")
        print(f"最佳分数: {self.training_history['best_score']:.4f}")
        print("\n生成的文件:")
        for file_type, filename in report['generated_files'].items():
            file_path = self.output_dir / filename
            if file_path.exists():
                print(f"  ✓ {file_type}: {filename}")
            else:
                print(f"  ✗ {file_type}: {filename} (文件不存在)")
        print("\n训练完成！")
        print("="*60)
        
        return report
        
    def run(self):
        """执行完整的训练流程"""
        try:
            # 1. 初始化组件
            self.setup_components()
            
            # 2. 加载数据
            self.load_data()
            
            # 3. 设置条件
            self.setup_conditions()
            
            # 4. 训练模型
            self.train_model()
            
            # 5. 生成序列
            generated_sequences = self.generate_sequences(
                n_sequences=self.args.n_generate
            )
            
            # 6. 评估序列
            self.evaluate_sequences(generated_sequences)
            
            # 7. 创建可视化
            self.create_visualizations()
            
            # 8. 生成最终报告
            final_report = self.generate_final_report()
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"训练过程发生错误: {e}")
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='DNA启动子优化系统 - 一键式训练脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础参数
    parser.add_argument('--quick-demo', action='store_true',
                       help='快速演示模式（使用生成的示例数据）')
    parser.add_argument('--data-path', type=str,
                       help='训练数据文件路径')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--embed-dim', type=int, default=256,
                       help='嵌入维度')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='隐藏层维度')
    parser.add_argument('--max-length', type=int, default=200,
                       help='序列最大长度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停耐心值')
    
    # 优化参数
    parser.add_argument('--mixed-precision', action='store_true',
                       help='启用混合精度训练')
    parser.add_argument('--distributed', action='store_true',
                       help='启用分布式训练')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                       help='梯度累积步数')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='梯度裁剪最大范数')
    
    # 数据参数
    parser.add_argument('--data-augmentation', action='store_true',
                       help='启用数据增强')
    parser.add_argument('--skip-data-clean', action='store_true',
                       help='跳过数据清洗')
    
    # 生成参数
    parser.add_argument('--n-generate', type=int, default=100,
                       help='生成序列数量')
    
    # 条件参数
    parser.add_argument('--conditions', type=str,
                       help='条件设置，格式：key1=value1,key2=value2')
    
    # 超参数调优
    parser.add_argument('--auto-tune', action='store_true',
                       help='启用自动超参数调优')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='超参数调优试验次数')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 打印启动信息
    print("\n" + "="*60)
    print("DNA启动子优化系统 - 一键式训练脚本")
    print("="*60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("="*60)
    
    # 参数验证
    if not args.quick_demo and not args.data_path:
        print("错误: 必须指定数据路径或使用--quick-demo模式")
        print("使用 --help 查看详细参数说明")
        sys.exit(1)
        
    # 创建训练器并运行
    trainer = OneClickTrainer(args)
    result = trainer.run()
    
    if result:
        print("\n训练成功完成！")
        sys.exit(0)
    else:
        print("\n训练失败！")
        sys.exit(1)


if __name__ == '__main__':
    main()