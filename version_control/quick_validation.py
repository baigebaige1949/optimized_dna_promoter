"""
版本控制系统快速验证脚本

验证核心功能是否正常工作
"""

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def quick_validation():
    """快速验证系统功能"""
    print("🚀 开始验证版本控制系统...")
    
    try:
        # 1. 测试导入
        print("\n1. 测试模块导入...")
        from version_control import (
            ModelVersionManager,
            PerformanceComparator, 
            ExperimentTracker,
            VisualizationManager
        )
        from version_control.integration import EnhancedTrainer
        from version_control.config import get_config
        print("✅ 所有模块导入成功")
        
        # 2. 测试配置系统
        print("\n2. 测试配置系统...")
        config = get_config("default")
        print(f"✅ 配置加载成功，基础目录: {config.base_dir}")
        
        # 3. 测试版本管理器初始化
        print("\n3. 测试版本管理器...")
        version_manager = ModelVersionManager("./test_versions")
        print("✅ 版本管理器初始化成功")
        
        # 4. 测试简单模型创建和保存
        print("\n4. 测试模型版本保存...")
        
        # 创建简单模型
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        test_model = TestModel()
        version_name = version_manager.save_version(
            test_model,
            "test_model_v1",
            description="快速验证测试模型"
        )
        print(f"✅ 模型版本保存成功: {version_name}")
        
        # 5. 测试版本列表
        print("\n5. 测试版本列表...")
        versions = version_manager.list_versions()
        print(f"✅ 版本列表获取成功，共 {len(versions)} 个版本")
        
        # 6. 测试实验跟踪器
        print("\n6. 测试实验跟踪器...")
        tracker = ExperimentTracker("./test_experiments")
        exp_id = tracker.start_experiment(
            "快速验证实验",
            "系统功能快速验证",
            tags=["validation"]
        )
        tracker.log_hyperparameters(exp_id, {"lr": 0.01, "batch_size": 16})
        tracker.log_metrics(exp_id, {"loss": 0.5, "accuracy": 0.8})
        tracker.end_experiment(exp_id, "completed")
        print(f"✅ 实验跟踪完成，实验ID: {exp_id[:8]}...")
        
        # 7. 测试集成训练器
        print("\n7. 测试增强训练器...")
        enhanced_trainer = EnhancedTrainer(
            "./test_training",
            auto_versioning=False,  # 关闭自动版本管理以简化测试
            auto_checkpoint=False
        )
        print("✅ 增强训练器初始化成功")
        
        # 8. 清理测试文件
        print("\n8. 清理测试文件...")
        import shutil
        test_dirs = ["./test_versions", "./test_experiments", "./test_training"]
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                shutil.rmtree(test_dir)
        print("✅ 测试文件清理完成")
        
        print("\n🎉 所有核心功能验证通过！")
        print("\n系统状态: 就绪 ✅")
        print("建议: 可以开始使用完整功能")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 验证过程中出现错误: {str(e)}")
        print("请检查依赖安装和代码完整性")
        return False

if __name__ == "__main__":
    print("="*60)
    print("版本控制和性能对比系统 - 快速验证")
    print("="*60)
    
    success = quick_validation()
    
    if success:
        print("\n" + "="*60)
        print("验证完成 - 系统就绪！")
        print("\n下一步:")
        print("1. 查看 usage_examples.py 了解详细用法")
        print("2. 查看 README.md 了解完整文档")
        print("3. 开始使用 EnhancedTrainer 进行模型训练")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("验证失败 - 需要检查系统")
        print("="*60)
        sys.exit(1)
