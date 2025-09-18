"""版本控制系统使用示例

展示如何使用版本管理、性能对比、实验跟踪等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import json
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from version_control import (
    ModelVersionManager, 
    PerformanceComparator, 
    ExperimentTracker, 
    VisualizationManager
)


class SimpleDNAModel(nn.Module):
    """简单的DNA序列预测模型（用于演示）"""
    
    def __init__(self, input_size=1000, hidden_size=128, num_classes=2):
        super(SimpleDNAModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ComplexDNAModel(nn.Module):
    """复杂的DNA序列预测模型（用于演示）"""
    
    def __init__(self, input_size=1000, hidden_size=256, num_classes=2):
        super(ComplexDNAModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc5 = nn.Linear(hidden_size // 4, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def create_mock_data(n_samples=1000, input_size=1000):
    """创建模拟数据"""
    X = torch.randn(n_samples, input_size)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(X, y)


def train_model_with_tracking(model, train_loader, val_loader, experiment_tracker, experiment_id, epochs=10):
    """带有实验跟踪的模型训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 记录超参数
    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': epochs,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'model_type': model.__class__.__name__
    }
    experiment_tracker.log_hyperparameters(experiment_id, hyperparams)
    
    training_history = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
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
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        # 计算指标
        train_acc = correct_predictions / total_samples
        val_acc = val_correct / val_total
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 记录指标
        metrics = {
            'train_loss': avg_train_loss,
            'train_accuracy': train_acc,
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc
        }
        
        experiment_tracker.log_metrics(experiment_id, metrics, epoch=epoch)
        training_history.append(metrics)
        
        # 记录训练日志
        log_msg = f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}"
        experiment_tracker.log_message(experiment_id, log_msg)
        print(log_msg)
    
    return training_history


def demo_version_control_workflow():
    """演示版本控制完整工作流程"""
    print("=" * 60)
    print("版本控制和性能对比系统演示")
    print("=" * 60)
    
    # 1. 初始化管理器
    print("\n1. 初始化管理器...")
    version_manager = ModelVersionManager("./demo_versions")
    performance_comparator = PerformanceComparator("./demo_results")
    experiment_tracker = ExperimentTracker("./demo_experiments")
    viz_manager = VisualizationManager("./demo_visualizations")
    
    # 2. 准备数据
    print("\n2. 准备数据...")
    train_data = create_mock_data(800, 1000)
    val_data = create_mock_data(200, 1000)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # 3. 训练简单模型
    print("\n3. 训练简单模型...")
    simple_model = SimpleDNAModel()
    exp1_id = experiment_tracker.start_experiment(
        "Simple_DNA_Model", 
        "简单DNA序列预测模型实验",
        tags=["baseline", "simple"]
    )
    
    train_history1 = train_model_with_tracking(
        simple_model, train_loader, val_loader, 
        experiment_tracker, exp1_id, epochs=5
    )
    
    # 保存模型版本
    version1 = version_manager.save_version(
        simple_model,
        "simple_dna_v1",
        metadata={"experiment_id": exp1_id, "final_val_acc": train_history1[-1]['val_accuracy']},
        description="简单DNA模型的第一个版本"
    )
    
    experiment_tracker.end_experiment(exp1_id, "completed")
    
    # 4. 训练复杂模型
    print("\n4. 训练复杂模型...")
    complex_model = ComplexDNAModel()
    exp2_id = experiment_tracker.start_experiment(
        "Complex_DNA_Model",
        "复杂DNA序列预测模型实验",
        tags=["enhanced", "complex"]
    )
    
    train_history2 = train_model_with_tracking(
        complex_model, train_loader, val_loader,
        experiment_tracker, exp2_id, epochs=5
    )
    
    # 保存模型版本
    version2 = version_manager.save_version(
        complex_model,
        "complex_dna_v1", 
        metadata={"experiment_id": exp2_id, "final_val_acc": train_history2[-1]['val_accuracy']},
        description="复杂DNA模型的第一个版本"
    )
    
    experiment_tracker.end_experiment(exp2_id, "completed")
    
    # 5. 性能对比
    print("\n5. 进行性能对比...")
    
    # 评估简单模型
    simple_results = performance_comparator.evaluate_model(
        simple_model, val_loader, nn.CrossEntropyLoss(),
        device='cpu', model_name="Simple_DNA_Model"
    )
    
    # 评估复杂模型
    complex_results = performance_comparator.evaluate_model(
        complex_model, val_loader, nn.CrossEntropyLoss(),
        device='cpu', model_name="Complex_DNA_Model"
    )
    
    # 比较模型
    comparison_result = performance_comparator.compare_models(
        [simple_results, complex_results],
        "DNA_Models_Comparison"
    )
    
    # 6. A/B测试
    print("\n6. 进行A/B测试...")
    ab_test_result = performance_comparator.ab_test(
        simple_results, complex_results
    )
    
    print(f"A/B测试结果: 推荐使用 {ab_test_result['summary']['recommendation']}")
    
    # 7. 生成报告
    print("\n7. 生成比较报告...")
    report_path = performance_comparator.generate_comparison_report(
        comparison_result,
        "./demo_results/comparison_report.md"
    )
    
    # 8. 可视化
    print("\n8. 生成可视化图表...")
    
    # 绘制训练曲线
    exp1_data = experiment_tracker.get_experiment(exp1_id)
    curve_path1 = viz_manager.plot_training_curves(
        exp1_data, 
        save_path="./demo_visualizations/simple_model_curves.png"
    )
    
    exp2_data = experiment_tracker.get_experiment(exp2_id) 
    curve_path2 = viz_manager.plot_training_curves(
        exp2_data,
        save_path="./demo_visualizations/complex_model_curves.png"
    )
    
    # 绘制指标对比
    comparison_path = viz_manager.plot_metric_comparison(
        [comparison_result], 
        "accuracy",
        save_path="./demo_visualizations/accuracy_comparison.png"
    )
    
    # 绘制性能雷达图
    radar_path = viz_manager.plot_performance_radar(
        comparison_result,
        save_path="./demo_visualizations/performance_radar.png"
    )
    
    # 绘制模型复杂度对比
    complexity_path = viz_manager.plot_model_complexity_comparison(
        [simple_results, complex_results],
        save_path="./demo_visualizations/complexity_comparison.png"
    )
    
    # 创建仪表盘
    dashboard_path = viz_manager.create_dashboard(
        exp1_data,
        [comparison_result],
        save_path="./demo_visualizations/dashboard.png"
    )
    
    # 9. 版本管理操作
    print("\n9. 版本管理操作...")
    
    # 列出所有版本
    versions = version_manager.list_versions()
    print(f"当前存储的模型版本数量: {len(versions)}")
    for version in versions:
        print(f"- {version['version_name']}: {version['description']}")
    
    # 比较版本
    if len(versions) >= 2:
        version_comparison = version_manager.compare_versions(
            versions[0]['version_name'], 
            versions[1]['version_name']
        )
        print(f"\n版本比较: {versions[0]['version_name']} vs {versions[1]['version_name']}")
        print(f"参数数量差异: {version_comparison['differences']['parameters_count']}")
    
    # 加载模型版本
    print("\n演示加载模型版本...")
    new_model = SimpleDNAModel()
    loaded_model = version_manager.load_model_weights(version1, new_model)
    print(f"成功加载版本: {version1}")
    
    # 10. 实验对比
    print("\n10. 实验对比...")
    experiment_comparison = experiment_tracker.compare_experiments(
        [exp1_id, exp2_id],
        ['train_accuracy', 'val_accuracy']
    )
    
    print(f"实验比较结果:")
    for metric, best in experiment_comparison['best_performers'].items():
        print(f"- {metric} 最佳: {best['experiment_name']} ({best['value']:.4f})")
    
    # 11. 导出结果
    print("\n11. 导出结果...")
    export_path = experiment_tracker.export_experiment(exp1_id, "./demo_results/experiment_export.json")
    
    print("\n演示完成！结果文件:")
    print(f"- 比较报告: {report_path}")
    print(f"- 训练曲线: {curve_path1}, {curve_path2}")
    print(f"- 指标对比: {comparison_path}")
    print(f"- 性能雷达图: {radar_path}")
    print(f"- 复杂度对比: {complexity_path}")
    print(f"- 仪表盘: {dashboard_path}")
    print(f"- 实验导出: {export_path}")
    
    return {
        'version_manager': version_manager,
        'performance_comparator': performance_comparator,
        'experiment_tracker': experiment_tracker,
        'visualization_manager': viz_manager,
        'comparison_result': comparison_result,
        'ab_test_result': ab_test_result
    }


def demo_model_rollback():
    """演示模型回滚功能"""
    print("\n" + "=" * 40)
    print("模型回滚功能演示")
    print("=" * 40)
    
    version_manager = ModelVersionManager("./demo_versions")
    
    # 列出可用版本
    versions = version_manager.list_versions()
    if len(versions) < 2:
        print("需要至少两个版本才能演示回滚功能")
        return
    
    print(f"可用的模型版本:")
    for i, version in enumerate(versions[:3]):  # 显示前3个
        print(f"{i+1}. {version['version_name']} - {version['description']}")
        print(f"   创建时间: {version['created_at']}")
    
    # 模拟生产环境中的模型部署
    current_version = versions[0]['version_name']  # 当前版本
    previous_version = versions[1]['version_name']  # 上一个版本
    
    print(f"\n当前生产环境使用版本: {current_version}")
    
    # 模拟版本问题（例如性能下降）
    print(f"\n检测到版本 {current_version} 存在性能问题，需要回滚...")
    
    # 执行回滚
    print(f"正在回滚到版本: {previous_version}")
    
    # 加载上一个版本
    rollback_model = SimpleDNAModel()
    loaded_model = version_manager.load_model_weights(previous_version, rollback_model)
    
    print(f"回滚成功！当前使用版本: {previous_version}")
    
    # 获取回滚版本信息
    rollback_info = version_manager.get_version_info(previous_version)
    print(f"回滚版本信息:")
    print(f"- 描述: {rollback_info['description']}")
    print(f"- 参数数量: {rollback_info['model_info']['parameters_count']}")
    print(f"- 创建时间: {rollback_info['created_at']}")


if __name__ == "__main__":
    # 执行主演示
    results = demo_version_control_workflow()
    
    # 执行回滚演示
    demo_model_rollback()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("请查看生成的文件和可视化结果")
    print("=" * 60)
