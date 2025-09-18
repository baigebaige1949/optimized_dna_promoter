#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级生成模块初始化文件
对外暴露所有生成相关的类和函数
"""

from .advanced_generation import (
    NoiseScheduler,
    CosineNoiseScheduler,
    LinearNoiseScheduler,
    QuadraticNoiseScheduler,
    Sampler,
    DPMSolverPlusPlusSampler,
    DDIMSampler,
    AbsorbEscapePostProcessor,
    AdvancedGenerationPipeline,
    create_generation_pipeline
)

__all__ = [
    # 噪声调度器
    'NoiseScheduler',
    'CosineNoiseScheduler',
    'LinearNoiseScheduler', 
    'QuadraticNoiseScheduler',
    
    # 采样器
    'Sampler',
    'DPMSolverPlusPlusSampler',
    'DDIMSampler',
    
    # 后处理器
    'AbsorbEscapePostProcessor',
    
    # 主要流水线
    'AdvancedGenerationPipeline',
    
    # 工厂函数
    'create_generation_pipeline'
]
