#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条件控制模块初始化文件
对外暴露所有条件控制相关的类和函数
"""

# 核心类
from .condition_controller import (
    ConditionType,
    ConditionRange,
    ConditionDefinition,
    ConditionVector,
    ConditionController,
    ConditionConstraint,
    RangeConstraint,
    BiologicalCompatibilityConstraint,
    create_condition_controller
)

from .intelligent_filling import (
    BiologicalKnowledge,
    KnowledgeBase,
    ConditionPredictor,
    CorrelationAnalyzer,
    IntelligentFiller,
    create_intelligent_filler
)

# 工厂函数
__all__ = [
    # 条件控制器相关
    'ConditionType',
    'ConditionRange', 
    'ConditionDefinition',
    'ConditionVector',
    'ConditionController',
    'ConditionConstraint',
    'RangeConstraint',
    'BiologicalCompatibilityConstraint',
    'create_condition_controller',
    
    # 智能填充相关
    'BiologicalKnowledge',
    'KnowledgeBase',
    'ConditionPredictor',
    'CorrelationAnalyzer',
    'IntelligentFiller',
    'create_intelligent_filler',
    
    # 快速创建接口
    'create_condition_system'
]

def create_condition_system():
    """
    创建完整的条件控制系统
    
    Returns:
        tuple: (condition_controller, intelligent_filler)
    """
    controller = create_condition_controller()
    filler = create_intelligent_filler()
    
    return controller, filler
