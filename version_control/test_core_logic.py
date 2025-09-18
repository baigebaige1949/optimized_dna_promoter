"""
版本控制系统核心逻辑验证

验证核心功能逻辑是否正确（不依赖外部库）
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

def test_core_logic():
    """测试核心逻辑功能"""
    print("🚀 开始核心逻辑验证...")
    
    try:
        # 1. 测试配置系统
        print("\n1. 测试配置系统...")
        from config import get_config, VersionControlConfig
        
        config = get_config("default")
        assert config.base_dir == "./version_control_workspace"
        assert config.auto_versioning == True
        print("✅ 配置系统验证通过")
        
        # 2. 测试实验跟踪器的核心逻辑
        print("\n2. 测试实验跟踪器...")
        from experiment_tracker import ExperimentTracker
        
        # 创建临时目录
        test_dir = "./test_experiments_core"
        tracker = ExperimentTracker(test_dir)
        
        # 测试实验创建
        exp_id = tracker.start_experiment(
            "核心逻辑测试",
            "测试描述",
            tags=["test"]
        )
        assert exp_id is not None
        assert len(exp_id) > 0
        
        # 测试参数记录
        test_params = {"lr": 0.001, "batch_size": 32}
        tracker.log_hyperparameters(exp_id, test_params)
        
        # 测试指标记录
        test_metrics = {"loss": 0.5, "accuracy": 0.85}
        tracker.log_metrics(exp_id, test_metrics, epoch=1)
        
        # 测试实验结束
        tracker.end_experiment(exp_id, "completed")
        
        # 验证实验数据
        exp_data = tracker.get_experiment(exp_id)
        assert exp_data['name'] == "核心逻辑测试"
        assert exp_data['status'] == "completed"
        assert "lr" in exp_data['hyperparameters']
        
        print("✅ 实验跟踪器核心逻辑验证通过")
        
        # 3. 测试版本管理器的核心逻辑
        print("\n3. 测试版本管理器...")
        from model_version_manager import ModelVersionManager
        
        # 创建临时目录
        version_dir = "./test_versions_core"
        version_manager = ModelVersionManager(version_dir)
        
        # 模拟模型信息（不使用真实的PyTorch模型）
        mock_model_info = {
            'class_name': 'MockModel',
            'model_type': 'MockModel',
            'parameters_count': 1000000,
            'trainable_parameters': 900000
        }
        
        # 模拟保存版本信息
        version_info = {
            'version_name': 'test_model_v1',
            'created_at': datetime.now().isoformat(),
            'description': '测试模型版本',
            'model_info': mock_model_info,
            'metadata': {'test': True},
        }
        
        # 直接测试版本信息管理
        version_manager.versions['test_model_v1'] = version_info
        version_manager._save_versions_index()
        
        # 测试版本列表
        versions = version_manager.list_versions()
        assert len(versions) > 0
        assert versions[0]['version_name'] == 'test_model_v1'
        
        print("✅ 版本管理器核心逻辑验证通过")
        
        # 4. 测试性能对比器的数据处理逻辑
        print("\n4. 测试性能对比器...")
        from performance_comparator import PerformanceComparator
        
        comparator = PerformanceComparator("./test_results_core")
        
        # 模拟模型评估结果
        mock_result1 = {
            'model_name': 'Model_A',
            'evaluation_time': datetime.now().isoformat(),
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.87,
                'f1_score': 0.85
            },
            'performance': {
                'avg_inference_time': 0.001,
                'samples_per_second': 1000
            },
            'model_complexity': {
                'total_parameters': 1000000,
                'model_size_mb': 10.0
            }
        }
        
        mock_result2 = {
            'model_name': 'Model_B',
            'evaluation_time': datetime.now().isoformat(),
            'metrics': {
                'accuracy': 0.87,
                'precision': 0.85,
                'recall': 0.89,
                'f1_score': 0.87
            },
            'performance': {
                'avg_inference_time': 0.002,
                'samples_per_second': 500
            },
            'model_complexity': {
                'total_parameters': 2000000,
                'model_size_mb': 20.0
            }
        }
        
        # 测试比较功能
        comparison = comparator.compare_models([mock_result1, mock_result2], "test_comparison")
        
        assert 'best_models' in comparison
        assert 'summary_table' in comparison
        assert len(comparison['summary_table']) == 2
        
        # 测试A/B测试
        ab_result = comparator.ab_test(mock_result1, mock_result2)
        assert 'summary' in ab_result
        assert 'recommendation' in ab_result['summary']
        
        print("✅ 性能对比器核心逻辑验证通过")
        
        # 5. 清理测试文件
        print("\n5. 清理测试文件...")
        import shutil
        test_dirs = [test_dir, version_dir, "./test_results_core"]
        for test_dir_path in test_dirs:
            if Path(test_dir_path).exists():
                shutil.rmtree(test_dir_path)
        print("✅ 测试文件清理完成")
        
        print("\n🎉 所有核心逻辑验证通过！")
        print("\n系统状态: 核心功能就绪 ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 核心逻辑验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("版本控制系统 - 核心逻辑验证")
    print("="*60)
    
    success = test_core_logic()
    
    if success:
        print("\n" + "="*60)
        print("核心逻辑验证成功 - 系统就绪！")
        print("\n注意: 完整功能需要安装 PyTorch, matplotlib, seaborn 等依赖")
        print("安装命令: pip install torch matplotlib seaborn pandas scikit-learn")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("核心逻辑验证失败")
        print("="*60)
        sys.exit(1)
