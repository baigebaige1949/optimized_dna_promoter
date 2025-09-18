# -*- coding: utf-8 -*-
"""第二阶段架构优化模块

提供多模态特征融合、高级训练、生物学评估和一键式流程的完整解决方案
"""

# 版本信息
__version__ = "2.0.0"
__author__ = "DNA Promoter Optimization Team"
__description__ = "第二阶段架构优化 - 多模态融合和高级训练框架"

# 导入主要组件
try:
    from .models.multimodal_fusion import (
        MultiModalPredictor,
        MultiModalFusionNetwork,
        CrossModalAttention,
        AdaptiveWeightFusion,
        create_multimodal_fusion_model
    )
except ImportError:
    pass

try:
    from .training.advanced_trainer import (
        AdvancedTrainer,
        create_advanced_trainer,
        distributed_training_context
    )
except ImportError:
    pass

try:
    from .evaluation.biological_metrics import (
        BiologicalMetrics,
        evaluate_generated_sequences
    )
except ImportError:
    pass

try:
    from .one_click_pipeline import OneClickPipeline
except ImportError:
    pass

# 公开API
__all__ = [
    # 多模态融合
    'MultiModalPredictor',
    'MultiModalFusionNetwork', 
    'CrossModalAttention',
    'AdaptiveWeightFusion',
    'create_multimodal_fusion_model',
    
    # 高级训练
    'AdvancedTrainer',
    'create_advanced_trainer',
    'distributed_training_context',
    
    # 生物学评估
    'BiologicalMetrics',
    'evaluate_generated_sequences',
    
    # 一键式流程
    'OneClickPipeline'
]

# 模块信息
MODULE_INFO = {
    'name': 'Stage2 Architecture Optimization',
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'components': {
        'multimodal_fusion': '多模态特征融合模块',
        'advanced_trainer': '高级训练模块', 
        'biological_metrics': '生物学评估模块',
        'one_click_pipeline': '一键式流程模块'
    },
    'features': [
        '跨模态注意力机制',
        '自适应权重融合',
        '分布式训练支持',
        '混合精度训练',
        '梯度裁剪和优化策略',
        'Jensen-Shannon散度评估',
        'S-FID生物学指标',
        'Motif分析和富集检测',
        '超参数自动调优',
        '综合可视化报告'
    ]
}

def get_module_info():
    """获取模块信息"""
    return MODULE_INFO

def print_module_info():
    """打印模块信息"""
    info = get_module_info()
    print(f"模块名称: {info['name']}")
    print(f"版本: {info['version']}")
    print(f"作者: {info['author']}")
    print(f"描述: {info['description']}")
    
    print("\n核心组件:")
    for component, desc in info['components'].items():
        print(f"  - {component}: {desc}")
        
    print("\n主要特性:")
    for feature in info['features']:
        print(f"  ✓ {feature}")
