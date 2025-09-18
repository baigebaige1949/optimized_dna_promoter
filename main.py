#!/usr/bin/env python3
"""
DNA启动子生成项目 - 优化版主程序
作者: MiniMax Agent
日期: 2025-08-12
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from optimized_dna_promoter.config import BaseConfig, ModelConfig, TrainingConfig
from optimized_dna_promoter.core import DiffusionModel, PromoterPredictor, FeatureExtractor
from optimized_dna_promoter.utils import setup_logging, get_logger, DeviceManager
from optimized_dna_promoter.utils import save_json, load_json, save_fasta, load_fasta


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="DNA启动子生成项目 - 优化版",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        choices=["demo", "generate", "train", "predict", "analyze"],
        default="demo",
        help="运行模式"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入文件路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="输出目录"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备 (auto, cpu, cuda)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="生成样本数量"
    )
    
    parser.add_argument(
        "--strength",
        choices=["weak", "medium", "strong"],
        default="medium",
        help="启动子强度"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不输出日志到控制台"
    )
    
    return parser.parse_args()


class DNAPromoterApp:
    """
DNA启动子生成应用主类
"""
    
    def __init__(self, config: BaseConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        
        # 初始化设备管理器
        self.device_manager = DeviceManager(config.device)
        
        # 初始化模型（延迟加载）
        self.diffusion_model = None
        self.predictor_model = None
        
        self.logger = get_logger(__name__)
        
    def _init_models(self) -> None:
        """初始化模型"""
        if self.diffusion_model is None:
            self.logger.info("初始化扩散模型...")
            self.diffusion_model = DiffusionModel(
                self.model_config.diffusion, 
                vocab_size=4
            )
            self.diffusion_model = self.device_manager.move_to_device(self.diffusion_model)
        
        if self.predictor_model is None:
            self.logger.info("初始化预测模型...")
            self.predictor_model = PromoterPredictor(self.model_config.predictor)
            self.predictor_model = self.device_manager.move_to_device(self.predictor_model)
    
    def demo_mode(self, args) -> None:
        """演示模式"""
        print("\n🧬 DNA启动子生成器 - 优化版演示")
        print("=" * 60)
        print("🔬 项目: 基于扩散模型的AI启动子设计工具")
        print("💻 开发者: MiniMax Agent")
        print("⏰ 日期: 2025-08-12 (优化版)")
        print("🎆 目标: 提供模块化、高效的DNA序列生成工具")
        print("=" * 60)
        
        # 显示系统信息
        print(f"\n📊 系统信息:")
        print(f"  计算设备: {self.device_manager.device}")
        print(f"  可用设备: {', '.join(self.device_manager.get_available_devices())}")
        
        memory_info = self.device_manager.get_memory_info()
        if memory_info:
            print(f"  内存信息: {memory_info}")
        
        # 初始化模型
        print(f"\n🚀 步骤1: 初始化模型...")
        self._init_models()
        print(f"  ✅ 扩散模型: {sum(p.numel() for p in self.diffusion_model.parameters()):,} 参数")
        print(f"  ✅ 预测模型: {sum(p.numel() for p in self.predictor_model.parameters()):,} 参数")
        
        # 演示特征提取
        print(f"\n🔬 步骤2: 特征提取演示...")
        sample_sequences = [
            "ATGCGATCGATCGATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTA",
            "TATATAAAGCGCGCGCATATATAGCGCG"
        ]
        
        feature_extractor = FeatureExtractor(self.model_config.predictor)
        features = feature_extractor.extract_features(sample_sequences)
        
        print(f"  ✅ 提取了 {len(sample_sequences)} 个序列的特征")
        for feat_name, feat_tensor in features.items():
            print(f"    {feat_name}: {feat_tensor.shape}")
        
        # 演示序列预测
        print(f"\n🎯 步骤3: 启动子强度预测...")
        predictions = self.predictor_model.predict_strength(sample_sequences)
        
        for i, (seq, pred) in enumerate(zip(sample_sequences, predictions), 1):
            print(f"  序列 {i}: {seq[:20]}... -> 强度: {pred:.3f}")
        
        # 演示生成（简化版）
        print(f"\n🧬 步骤4: 序列生成演示 (简化版)...")
        print(f"  正在生成 {args.num_samples} 个序列...")
        
        # 简化的模拟生成
        generated_sequences = self._simulate_generation(args.num_samples, args.strength)
        
        print(f"  ✅ 已生成 {len(generated_sequences)} 个序列")
        
        # 保存结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存FASTA文件
        fasta_sequences = {
            f"generated_{i+1}_{args.strength}": seq 
            for i, seq in enumerate(generated_sequences)
        }
        
        fasta_path = output_path / f"demo_generated_{args.strength}.fasta"
        save_fasta(fasta_sequences, fasta_path)
        
        # 保存结果统计
        results = {
            "generated_count": len(generated_sequences),
            "target_strength": args.strength,
            "sequences": list(generated_sequences),
            "predictions": predictions,
            "device_info": str(self.device_manager),
            "model_info": {
                "diffusion_params": sum(p.numel() for p in self.diffusion_model.parameters()),
                "predictor_params": sum(p.numel() for p in self.predictor_model.parameters())
            }
        }
        
        results_path = output_path / "demo_results.json"
        save_json(results, results_path)
        
        print(f"\n🎆 演示完成!")
        print(f"  FASTA文件: {fasta_path}")
        print(f"  结果文件: {results_path}")
        print(f"  输出目录: {output_path}")
    
    def _simulate_generation(self, num_samples: int, strength: str) -> List[str]:
        """模拟序列生成（演示用）"""
        import random
        import string
        
        sequences = []
        nucleotides = ['A', 'T', 'G', 'C']
        
        # 根据强度调整GC含量
        gc_content_map = {
            'weak': 0.3,
            'medium': 0.5, 
            'strong': 0.7
        }
        
        target_gc = gc_content_map.get(strength, 0.5)
        
        for _ in range(num_samples):
            sequence = []
            for _ in range(200):  # 生成200bp的序列
                if random.random() < target_gc:
                    sequence.append(random.choice(['G', 'C']))
                else:
                    sequence.append(random.choice(['A', 'T']))
            
            sequences.append(''.join(sequence))
        
        return sequences
    
    def generate_mode(self, args) -> None:
        """生成模式"""
        self.logger.info(f"开始生成模式 - 生成 {args.num_samples} 个 {args.strength} 强度的启动子")
        
        self._init_models()
        
        # 这里实现真正的生成逻辑
        # 目前使用模拟
        generated_sequences = self._simulate_generation(args.num_samples, args.strength)
        
        # 保存结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fasta_sequences = {
            f"generated_{i+1}_{args.strength}": seq 
            for i, seq in enumerate(generated_sequences)
        }
        
        fasta_path = output_path / f"generated_{args.strength}_{args.num_samples}.fasta"
        save_fasta(fasta_sequences, fasta_path)
        
        self.logger.info(f"生成完成，结果保存在: {fasta_path}")
    
    def predict_mode(self, args) -> None:
        """预测模式"""
        if not args.input:
            raise ValueError("预测模式需要指定输入文件")
        
        self.logger.info(f"开始预测模式 - 分析文件: {args.input}")
        
        # 加载序列
        sequences_dict = load_fasta(args.input)
        sequences = list(sequences_dict.values())
        headers = list(sequences_dict.keys())
        
        self._init_models()
        
        # 预测强度
        predictions = self.predictor_model.predict_strength(sequences)
        
        # 结果分析
        results = []
        for header, seq, pred in zip(headers, sequences, predictions):
            results.append({
                'header': header,
                'sequence': seq,
                'predicted_strength': pred,
                'length': len(seq),
                'gc_content': (seq.count('G') + seq.count('C')) / len(seq)
            })
        
        # 保存结果
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_path = output_path / "prediction_results.json"
        save_json(results, results_path)
        
        self.logger.info(f"预测完成，结果保存在: {results_path}")
        
        # 显示统计信息
        avg_strength = sum(predictions) / len(predictions)
        print(f"\n统计信息:")
        print(f"  总序列数: {len(sequences)}")
        print(f"  平均强度: {avg_strength:.3f}")
        print(f"  强度范围: {min(predictions):.3f} - {max(predictions):.3f}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(
        log_level=args.log_level,
        log_dir="logs",
        log_to_console=not args.quiet
    )
    
    logger = get_logger(__name__)
    logger.info(f"DNA启动子生成器启动 - 模式: {args.mode}")
    
    try:
        # 初始化配置
        if args.config:
            base_config = BaseConfig.from_yaml(args.config)
        else:
            base_config = BaseConfig()
            base_config.device = args.device
            base_config.batch_size = args.batch_size
        
        model_config = ModelConfig()
        
        # 创建应用
        app = DNAPromoterApp(base_config, model_config)
        
        # 运行指定模式
        if args.mode == "demo":
            app.demo_mode(args)
        elif args.mode == "generate":
            app.generate_mode(args)
        elif args.mode == "predict":
            app.predict_mode(args)
        elif args.mode == "train":
            logger.error("训练模式尚未实现")
            return 1
        elif args.mode == "analyze":
            logger.error("分析模式尚未实现")
            return 1
        else:
            logger.error(f"未知模式: {args.mode}")
            return 1
        
        logger.info("程序执行完成")
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        return 130
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
