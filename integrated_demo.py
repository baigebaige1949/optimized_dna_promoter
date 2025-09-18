"""
Dirichlet扩散和版本控制系统的完整集成和演示
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Any
import os
from pathlib import Path

# 导入我们的模块
from models.dirichlet_diffusion import DirichletDiffusionModel, create_dirichlet_model
from version_control.model_version_manager import ModelVersionManager
from version_control.performance_comparator import PerformanceComparator
from version_control.visualization import VersionControlVisualizer
from version_control.experiment_tracker import ExperimentTracker


class IntegratedDirichletVersionControl:
    """整合Dirichlet扩散和版本控制的主类"""
    
    def __init__(self, workspace_dir: str = "./integrated_workspace"):
        """初始化集成系统
        
        Args:
            workspace_dir: 工作空间目录
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化各个组件
        self.version_manager = ModelVersionManager(str(self.workspace_dir / "versions"))
        self.performance_comparator = PerformanceComparator(str(self.workspace_dir / "results"))
        self.visualizer = VersionControlVisualizer(str(self.workspace_dir / "visualizations"))
        self.experiment_tracker = ExperimentTracker(str(self.workspace_dir / "experiments"))
        
        # 设备管理
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def create_sample_data(self, batch_size: int = 100, sequence_length: int = 50) -> DataLoader:
        """创建示例DNA数据
        
        Args:
            batch_size: 批次大小
            sequence_length: 序列长度
            
        Returns:
            数据加载器
        """
        # 生成随机DNA序列数据 (4维one-hot编码)
        data = torch.randn(batch_size, sequence_length, 4)
        data = torch.softmax(data, dim=-1)  # 转换为概率分布
        
        # 生成随机标签（示例）
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        return dataloader
    
    def train_model_with_tracking(self, 
                                model: DirichletDiffusionModel,
                                train_loader: DataLoader,
                                val_loader: DataLoader,
                                epochs: int = 10,
                                model_name: str = "dirichlet_model",
                                experiment_name: str = None) -> Dict[str, Any]:
        """带跟踪的模型训练
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            model_name: 模型名称
            experiment_name: 实验名称
            
        Returns:
            训练结果
        """
        # 开始实验跟踪
        if experiment_name is None:
            experiment_name = f"{model_name}_training"
        
        experiment_id = self.experiment_tracker.start_experiment(
            name=experiment_name,
            description=f"Training {model_name} for {epochs} epochs",
            config={
                'model_type': 'DirichletDiffusionModel',
                'epochs': epochs,
                'device': str(self.device),
                'sequence_length': model.sequence_length,
                'hidden_dim': model.hidden_dim
            }
        )
        
        # 记录超参数
        hyperparams = {
            'learning_rate': 0.001,
            'batch_size': train_loader.batch_size,
            'optimizer': 'Adam'
        }
        self.experiment_tracker.log_hyperparameters(hyperparams, experiment_id)
        
        # 初始化训练组件
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 训练历史
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': list(range(1, epochs + 1))
        }
        
        self.experiment_tracker.add_log(f"Starting training for {epochs} epochs", 'info', experiment_id)
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_losses = []
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 简化的损失计算（实际中应使用model.compute_loss）
                # 这里为了演示使用简单的重构损失
                t = torch.rand((data.size(0), 1), device=self.device)
                
                # 转换为stick-breaking参数
                from models.dirichlet_diffusion import StickBreakingTransform
                sb_transform = StickBreakingTransform()
                stick_breaking = sb_transform.simplex_to_stick_breaking(data)
                
                # 前向传播
                output = model(stick_breaking, t)
                loss = criterion(output, stick_breaking)  # 简化损失
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            training_history['train_loss'].append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    t = torch.rand((data.size(0), 1), device=self.device)
                    stick_breaking = sb_transform.simplex_to_stick_breaking(data)
                    output = model(stick_breaking, t)
                    loss = criterion(output, stick_breaking)
                    
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            training_history['val_loss'].append(avg_val_loss)
            
            # 记录指标
            self.experiment_tracker.log_metrics({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch + 1
            }, step=epoch + 1, experiment_id=experiment_id)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 结束实验
        self.experiment_tracker.add_log("Training completed successfully", 'info', experiment_id)
        self.experiment_tracker.end_experiment(experiment_id, 'completed')
        
        # 保存模型版本
        version_name = self.version_manager.save_version(
            model=model,
            version_name=f"{model_name}_epoch_{epochs}",
            description=f"Trained for {epochs} epochs with final val_loss: {avg_val_loss:.4f}",
            metadata={
                'experiment_id': experiment_id,
                'final_train_loss': avg_train_loss,
                'final_val_loss': avg_val_loss,
                'epochs_trained': epochs
            }
        )
        
        # 保存训练历史
        history_path = self.workspace_dir / f"{model_name}_history.json"
        import json
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=2, ensure_ascii=False, default=str)
        
        # 将训练历史保存为实验产物
        self.experiment_tracker.save_artifact(str(history_path), f"{model_name}_training_history.json", experiment_id)
        
        return {
            'version_name': version_name,
            'experiment_id': experiment_id,
            'training_history': training_history,
            'final_metrics': {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
        }
    
    def run_comprehensive_demo(self) -> Dict[str, Any]:
        """运行综合演示
        
        Returns:
            演示结果
        """
        print("=" * 50)
        print("Dirichlet扩散和版本控制系统综合演示")
        print("=" * 50)
        
        # 1. 创建示例数据
        print("\n1. 创建示例数据...")
        train_loader = self.create_sample_data(batch_size=200, sequence_length=50)
        val_loader = self.create_sample_data(batch_size=100, sequence_length=50)
        test_loader = self.create_sample_data(batch_size=100, sequence_length=50)
        
        # 2. 创建不同配置的模型
        print("\n2. 创建不同配置的模型...")
        
        models_config = [
            {
                'name': 'small_model',
                'sequence_length': 50,
                'hidden_dim': 128,
                'num_layers': 4,
                'num_heads': 4
            },
            {
                'name': 'medium_model',
                'sequence_length': 50,
                'hidden_dim': 256,
                'num_layers': 6,
                'num_heads': 8
            },
            {
                'name': 'large_model',
                'sequence_length': 50,
                'hidden_dim': 512,
                'num_layers': 8,
                'num_heads': 16
            }
        ]
        
        # 3. 训练模型并跟踪结果
        print("\n3. 训练模型并跟踪结果...")
        training_results = []
        model_evaluation_results = []
        
        for config in models_config:
            print(f"\n训练 {config['name']}...")
            
            # 创建模型
            model = create_dirichlet_model(**config)
            
            # 训练模型
            training_result = self.train_model_with_tracking(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=5,  # 简化演示，使用少量轮数
                model_name=config['name']
            )
            
            training_results.append({
                'config': config,
                'training_result': training_result
            })
            
            # 评估模型性能
            print(f"评估 {config['name']} 性能...")
            criterion = nn.MSELoss()
            
            eval_result = self.performance_comparator.evaluate_model(
                model=model,
                data_loader=test_loader,
                criterion=criterion,
                device=str(self.device),
                model_name=config['name']
            )
            
            model_evaluation_results.append(eval_result)
        
        # 4. 比较模型性能
        print("\n4. 比较模型性能...")
        comparison_result = self.performance_comparator.compare_models(
            model_evaluation_results,
            comparison_name="dirichlet_models_comparison"
        )
        
        # 5. 生成可视化报告
        print("\n5. 生成可视化报告...")
        
        # 准备实验数据用于可视化
        experiments_data = []
        for result in training_results:
            experiments_data.append({
                'name': result['config']['name'],
                'metrics_history': {
                    'loss': result['training_result']['training_history']['train_loss'],
                    'val_loss': result['training_result']['training_history']['val_loss']
                }
            })
        
        visualization_plots = self.visualizer.create_comprehensive_report(
            comparison_data=comparison_result,
            experiments_data=experiments_data,
            report_name="dirichlet_comprehensive_report"
        )
        
        # 6. 生成最终报告
        print("\n6. 生成最终报告...")
        final_report = self._generate_final_report(
            training_results=training_results,
            comparison_result=comparison_result,
            visualization_plots=visualization_plots
        )
        
        print("\n=" * 50)
        print("演示完成！")
        print(f"结果保存在: {self.workspace_dir}")
        print("=" * 50)
        
        return final_report
    
    def _generate_final_report(self, 
                             training_results: List[Dict],
                             comparison_result: Dict[str, Any],
                             visualization_plots: Dict[str, str]) -> Dict[str, Any]:
        """生成最终报告
        
        Args:
            training_results: 训练结果
            comparison_result: 比较结果
            visualization_plots: 可视化图表
            
        Returns:
            最终报告
        """
        report = {
            'demo_summary': {
                'total_models_trained': len(training_results),
                'best_performing_model': comparison_result.get('best_models', {}).get('overall_best', 'N/A'),
                'workspace_directory': str(self.workspace_dir)
            },
            'training_results': training_results,
            'performance_comparison': comparison_result,
            'visualization_plots': visualization_plots,
            'generated_files': {
                'model_versions': len(self.version_manager.list_versions()),
                'experiments': len(self.experiment_tracker.list_experiments()),
                'comparison_reports': 1,
                'visualization_plots': len(visualization_plots)
            }
        }
        
        # 保存最终报告
        report_file = self.workspace_dir / "comprehensive_demo_report.json"
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return report
    
    def quick_demo(self) -> Dict[str, Any]:
        """快速演示版本
        
        Returns:
            快速演示结果
        """
        print("快速演示Dirichlet扩散和版本控制...")
        
        # 创建小模型进行快速演示
        model = create_dirichlet_model(
            sequence_length=30,
            hidden_dim=64,
            num_layers=2,
            num_heads=4
        )
        
        # 创建小量数据
        train_loader = self.create_sample_data(batch_size=50, sequence_length=30)
        val_loader = self.create_sample_data(batch_size=30, sequence_length=30)
        
        # 快速训练
        training_result = self.train_model_with_tracking(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            model_name="quick_demo_model",
            experiment_name="quick_demo"
        )
        
        print(f"\n快速演示完成！")
        print(f"模型版本: {training_result['version_name']}")
        print(f"实验ID: {training_result['experiment_id']}")
        print(f"最终损失: {training_result['final_metrics']['val_loss']:.4f}")
        
        return training_result


def run_integrated_demo(demo_type: str = "comprehensive") -> Dict[str, Any]:
    """运行集成演示
    
    Args:
        demo_type: 演示类型 ('comprehensive' 或 'quick')
        
    Returns:
        演示结果
    """
    # 创建集成系统
    system = IntegratedDirichletVersionControl()
    
    if demo_type == "comprehensive":
        return system.run_comprehensive_demo()
    else:
        return system.quick_demo()


if __name__ == "__main__":
    # 运行快速演示
    result = run_integrated_demo(demo_type="quick")
    print("\n演示结果:", result)
