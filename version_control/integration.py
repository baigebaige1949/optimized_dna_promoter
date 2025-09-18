"""与训练系统的集成模块

将版本控制功能集成到现有的训练流程中
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from version_control import (
    ModelVersionManager, 
    PerformanceComparator, 
    ExperimentTracker, 
    VisualizationManager
)


class EnhancedTrainer:
    """增强的训练器，集成版本控制和性能对比功能"""
    
    def __init__(self, 
                 base_dir: str = "./training_workspace",
                 auto_versioning: bool = True,
                 auto_checkpoint: bool = True,
                 checkpoint_interval: int = 10):
        """初始化增强训练器
        
        Args:
            base_dir: 基础工作目录
            auto_versioning: 是否自动版本管理
            auto_checkpoint: 是否自动保存检查点
            checkpoint_interval: 检查点保存间隔轮数
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        self.auto_versioning = auto_versioning
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        
        # 初始化管理器
        self.version_manager = ModelVersionManager(str(self.base_dir / "versions"))
        self.performance_comparator = PerformanceComparator(str(self.base_dir / "results"))
        self.experiment_tracker = ExperimentTracker(str(self.base_dir / "experiments"))
        self.viz_manager = VisualizationManager(str(self.base_dir / "visualizations"))
        
        # 当前实验状态
        self.current_experiment_id = None
        self.training_history = []
        
    def start_experiment(self, 
                        experiment_name: str,
                        description: str = "",
                        tags: List[str] = None,
                        hyperparameters: Dict[str, Any] = None):
        """开始新实验
        
        Args:
            experiment_name: 实验名称
            description: 实验描述
            tags: 实验标签
            hyperparameters: 超参数
        """
        self.current_experiment_id = self.experiment_tracker.start_experiment(
            experiment_name, description, tags
        )
        
        if hyperparameters:
            self.experiment_tracker.log_hyperparameters(
                self.current_experiment_id, hyperparameters
            )
        
        self.training_history = []
        print(f"Started experiment: {experiment_name} (ID: {self.current_experiment_id})")
    
    def train_model(self,
                   model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epochs: int,
                   device: str = 'cpu',
                   model_name: str = None) -> Dict[str, Any]:
        """训练模型，集成版本控制功能
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            epochs: 训练轮数
            device: 计算设备
            model_name: 模型名称
            
        Returns:
            训练结果
        """
        if not self.current_experiment_id:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        model_name = model_name or model.__class__.__name__
        model = model.to(device)
        
        # 记录训练开始
        self.experiment_tracker.log_message(
            self.current_experiment_id, 
            f"Started training {model_name} for {epochs} epochs"
        )
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # 验证阶段
            model.eval()
            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion, device
            )
            
            # 记录指标
            metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            
            self.experiment_tracker.log_metrics(
                self.current_experiment_id, metrics, epoch=epoch
            )
            self.training_history.append(metrics)
            
            # 输出进度
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # 自动保存检查点
            if self.auto_checkpoint and (epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_name = self.version_manager.create_checkpoint(
                    model, optimizer, epoch, val_loss, metrics
                )
                self.experiment_tracker.log_artifact(
                    self.current_experiment_id,
                    f"checkpoint_epoch_{epoch+1}",
                    checkpoint_name,
                    "checkpoint"
                )
            
            # 记录最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                if self.auto_versioning:
                    version_name = f"{model_name}_best_epoch_{epoch+1}"
                    saved_version = self.version_manager.save_version(
                        model,
                        version_name,
                        metadata={
                            'experiment_id': self.current_experiment_id,
                            'epoch': epoch + 1,
                            'val_accuracy': val_acc,
                            'val_loss': val_loss
                        },
                        description=f"第{epoch+1}轮训练的最佳模型",
                        auto_increment=True
                    )
                    
                    self.experiment_tracker.log_artifact(
                        self.current_experiment_id,
                        "best_model_version",
                        saved_version,
                        "model_version"
                    )
        
        # 训练完成
        training_result = {
            'model_name': model_name,
            'epochs': epochs,
            'best_val_accuracy': best_val_acc,
            'training_history': self.training_history,
            'final_metrics': self.training_history[-1] if self.training_history else {}
        }
        
        self.experiment_tracker.log_message(
            self.current_experiment_id,
            f"Training completed. Best validation accuracy: {best_val_acc:.4f}"
        )
        
        return training_result
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, device):
        """训练一个 epoch"""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_predictions += (pred == target).sum().item()
            total_samples += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model, val_loader, criterion, device):
        """验证一个 epoch"""
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct_predictions += (pred == target).sum().item()
                total_samples += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def end_experiment(self, status: str = "completed"):
        """结束当前实验"""
        if self.current_experiment_id:
            self.experiment_tracker.end_experiment(self.current_experiment_id, status)
            
            # 生成训练曲线
            exp_data = self.experiment_tracker.get_experiment(self.current_experiment_id)
            curve_path = self.viz_manager.plot_training_curves(exp_data)
            
            self.experiment_tracker.log_artifact(
                self.current_experiment_id,
                "training_curves",
                curve_path,
                "visualization"
            )
            
            print(f"Experiment {self.current_experiment_id} ended with status: {status}")
            self.current_experiment_id = None
    
    def evaluate_and_compare_models(self, 
                                   models_and_loaders: List[tuple],
                                   comparison_name: str = None) -> Dict[str, Any]:
        """评估和比较多个模型
        
        Args:
            models_and_loaders: [(model, dataloader, model_name), ...] 列表
            comparison_name: 比較名称
            
        Returns:
            比较结果
        """
        if comparison_name is None:
            comparison_name = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_results = []
        criterion = nn.CrossEntropyLoss()
        
        for model, dataloader, model_name in models_and_loaders:
            print(f"Evaluating {model_name}...")
            
            result = self.performance_comparator.evaluate_model(
                model, dataloader, criterion, 
                device='cpu', model_name=model_name
            )
            model_results.append(result)
        
        # 比较模型
        comparison_result = self.performance_comparator.compare_models(
            model_results, comparison_name
        )
        
        # 生成报告
        report_path = self.performance_comparator.generate_comparison_report(
            comparison_result
        )
        
        # 生成可视化
        viz_paths = {
            'metric_comparison': self.viz_manager.plot_metric_comparison(
                [comparison_result], 'accuracy'
            ),
            'performance_radar': self.viz_manager.plot_performance_radar(
                comparison_result
            ),
            'complexity_comparison': self.viz_manager.plot_model_complexity_comparison(
                model_results
            )
        }
        
        print(f"Model comparison completed. Results saved to {report_path}")
        
        return {
            'comparison_result': comparison_result,
            'report_path': report_path,
            'visualizations': viz_paths
        }
    
    def rollback_to_version(self, version_name: str, target_model: nn.Module) -> nn.Module:
        """回滚到指定版本
        
        Args:
            version_name: 目标版本名称
            target_model: 要加载权重的模型
            
        Returns:
            加载权重后的模型
        """
        print(f"Rolling back to version: {version_name}")
        
        # 记录回滚操作
        if self.current_experiment_id:
            self.experiment_tracker.log_message(
                self.current_experiment_id,
                f"Rolled back to version: {version_name}",
                level="warning"
            )
        
        # 加载版本
        loaded_model = self.version_manager.load_model_weights(version_name, target_model)
        
        # 获取版本信息
        version_info = self.version_manager.get_version_info(version_name)
        print(f"Rollback completed. Version info:")
        print(f"- Description: {version_info['description']}")
        print(f"- Created: {version_info['created_at']}")
        print(f"- Parameters: {version_info['model_info']['parameters_count']}")
        
        return loaded_model
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """获取实验摘要信息"""
        if not self.current_experiment_id:
            return {"error": "No active experiment"}
        
        exp_data = self.experiment_tracker.get_experiment(self.current_experiment_id)
        
        # 计算统计信息
        if 'metric_history' in exp_data and exp_data['metric_history']:
            history = exp_data['metric_history']
            val_accuracies = [entry['metrics'].get('val_accuracy', 0) 
                            for entry in history 
                            if entry['metrics'].get('val_accuracy') is not None]
            
            summary = {
                'experiment_id': self.current_experiment_id,
                'experiment_name': exp_data['name'],
                'status': exp_data['status'],
                'epochs_completed': len(history),
                'best_val_accuracy': max(val_accuracies) if val_accuracies else 0,
                'latest_metrics': history[-1]['metrics'] if history else {},
                'hyperparameters': exp_data.get('hyperparameters', {})
            }
        else:
            summary = {
                'experiment_id': self.current_experiment_id,
                'experiment_name': exp_data['name'],
                'status': exp_data['status'],
                'epochs_completed': 0,
                'hyperparameters': exp_data.get('hyperparameters', {})
            }
        
        return summary
    
    def list_available_versions(self) -> List[Dict[str, Any]]:
        """列出可用的模型版本"""
        return self.version_manager.list_versions()
    
    def cleanup_old_versions(self, keep_count: int = 10):
        """清理旧版本，保留最新的几个
        
        Args:
            keep_count: 保留的版本数量
        """
        versions = self.version_manager.list_versions()
        
        if len(versions) > keep_count:
            versions_to_delete = versions[keep_count:]
            
            print(f"Cleaning up {len(versions_to_delete)} old versions...")
            
            for version in versions_to_delete:
                try:
                    self.version_manager.delete_version(
                        version['version_name'], confirm=True
                    )
                    print(f"Deleted version: {version['version_name']}")
                except Exception as e:
                    print(f"Failed to delete version {version['version_name']}: {e}")
