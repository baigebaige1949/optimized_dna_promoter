#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三阶段系统完善验证脚本
快速验证所有组件的正确性和可用性
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """测试所有模块导入"""
    print("📦 测试模块导入...")
    
    try:
        # 测试条件控制模块
        from optimized_dna_promoter.conditions import (
            create_condition_system, ConditionType, ConditionVector,
            create_condition_controller, create_intelligent_filler
        )
        print("✅ 条件控制模块导入成功")
        
        # 测试生成模块
        from optimized_dna_promoter.generation import (
            create_generation_pipeline, AdvancedGenerationPipeline,
            CosineNoiseScheduler, DPMSolverPlusPlusSampler
        )
        print("✅ 生成模块导入成功")
        
        # 测试数据处理模块
        from optimized_dna_promoter.data import (
            create_enhanced_dataset, EnhancedDataset, DataFormat,
            SequenceValidator, DataAugmentor
        )
        print("✅ 数据处理模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        traceback.print_exc()
        return False

def test_condition_system():
    """测试条件系统"""
    print("\n🎯 测试条件系统...")
    
    try:
        from optimized_dna_promoter.conditions import create_condition_system, ConditionType
        
        # 创建系统
        controller, filler = create_condition_system()
        
        # 测试条件创建
        conditions = controller.create_condition_vector({
            ConditionType.TEMPERATURE: 37.0,
            ConditionType.PH: 7.0
        })
        
        # 测试填充
        filled = filler.intelligent_fill(conditions, 'e_coli')
        
        # 测试验证
        is_valid, errors = controller.validate_conditions(filled)
        
        print(f"✅ 条件系统测试成功: {len(filled)}个条件, 有效: {is_valid}")
        return True
        
    except Exception as e:
        print(f"❌ 条件系统测试失败: {e}")
        return False

def test_generation_pipeline():
    """测试生成流水线"""
    print("\n✨ 测试生成流水线...")
    
    try:
        from optimized_dna_promoter.generation import create_generation_pipeline
        
        # 创建流水线
        pipeline = create_generation_pipeline({
            'noise_scheduler': 'cosine',
            'sampler': 'ddim',
            'post_process': True
        })
        
        # 测试噪声调度器
        schedule = pipeline.noise_scheduler.get_schedule(50)
        
        # 测试后处理器
        import torch
        mock_sequences = torch.randn(2, 10, 4)
        processed = pipeline.post_processor.process(mock_sequences)
        
        print(f"✅ 生成流水线测试成功: 调度长度 {len(schedule)}, 后处理形状 {processed.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 生成流水线测试失败: {e}")
        return False

def test_enhanced_dataset():
    """测试增强数据集"""
    print("\n📈 测试增强数据集...")
    
    try:
        from optimized_dna_promoter.data import create_enhanced_dataset
        
        # 创建数据集
        dataset = create_enhanced_dataset(max_length=50, vocab_size=4)
        
        # 测试数据添加
        sequences = ['ATCG' * 10, 'GCTA' * 8, 'CGAT' * 12]
        labels = [1.0, 2.0, 1.5]
        dataset.add_sequences(sequences, labels)
        
        # 测试质量检查
        quality = dataset.quality_check()
        
        # 测试数据编码
        encoded = dataset.encode_sequences(sequences)
        decoded = dataset.decode_sequences(encoded)
        
        print(f"✅ 增强数据集测试成功: {quality['total_sequences']}条序列, 编码形状 {encoded.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 增强数据集测试失败: {e}")
        return False

def test_integration():
    """测试组件集成"""
    print("\n🔗 测试组件集成...")
    
    try:
        from optimized_dna_promoter.conditions import create_condition_system, ConditionType
        from optimized_dna_promoter.generation import create_generation_pipeline
        from optimized_dna_promoter.data import create_enhanced_dataset
        
        # 创建所有组件
        controller, filler = create_condition_system()
        pipeline = create_generation_pipeline({'noise_scheduler': 'linear'})
        dataset = create_enhanced_dataset()
        
        # 测试数据流
        conditions = controller.create_condition_vector({ConditionType.TEMPERATURE: 30.0})
        filled_conditions = filler.intelligent_fill(conditions, 'yeast')
        
        # 添加测试数据
        dataset.add_sequences(['ATCGATCG', 'GCTAGCTA'], [1.0, 2.0])
        stats = dataset.get_statistics()
        
        print(f"✅ 组件集成测试成功: {len(filled_conditions)}个条件, {stats['total_sequences']}条序列")
        return True
        
    except Exception as e:
        print(f"❌ 组件集成测试失败: {e}")
        return False

def main():
    """主验证流程"""
    print("🚀 第三阶段系统完善验证开始")
    print("="*50)
    
    tests = [
        ("模块导入", test_imports),
        ("条件系统", test_condition_system),
        ("生成流水线", test_generation_pipeline),
        ("增强数据集", test_enhanced_dataset),
        ("组件集成", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}测试发生异常: {e}")
            results.append((test_name, False))
    
    # 统计结果
    print("\n" + "="*50)
    print("📄 验证结果汇总")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:<15} {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\n总计: {passed}/{total} 项测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✨ 所有组件验证通过，第三阶段系统完善成功！")
        return True
    else:
        print("⚠️ 部分组件验证失败，请检查错误信息")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
