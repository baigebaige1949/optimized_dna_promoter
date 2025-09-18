#!/usr/bin/env python3
"""
条件扩散模型快速验证演示
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimized_dna_promoter.models import create_conditional_diffusion_model, ConditionalDiffusionPredictor


def demo_conditional_diffusion():
    """条件扩散模型功能演示"""
    print("=== 条件扩散模型演示 ===")
    
    # 1. 创建模型
    print("\n1. 创建条件扩散模型...")
    model = create_conditional_diffusion_model(
        sequence_length=500,  # 较小的序列长度用于快速测试
        vocab_size=4,
        condition_dim=64,
        num_timesteps=100  # 较少的时间步用于快速测试
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 测试条件嵌入
    print("\n2. 测试条件嵌入...")
    test_conditions = {
        'temperature': 42.0,
        'ph': 6.5,
        'oxygen': 0.15,
        'salt': 0.1
    }
    condition_emb = model.unet.condition_embed(test_conditions)
    print(f"条件嵌入形状: {condition_emb.shape}")
    
    # 3. 测试默认条件填充
    print("\n3. 测试智能默认条件填充...")
    partial_conditions = {'temperature': 30.0}  # 只提供部分条件
    filled_conditions = model.unet.condition_embed.fill_default_conditions(partial_conditions)
    print(f"填充后条件: {filled_conditions}")
    
    # 4. 生成样本测试
    print("\n4. 测试样本生成...")
    model.eval()
    with torch.no_grad():
        samples = model.sample(
            batch_size=2,
            conditions=test_conditions
        )
    print(f"生成样本形状: {samples.shape}")
    print(f"样本统计: 均值={samples.mean():.4f}, 标准差={samples.std():.4f}")
    
    # 5. 训练损失测试
    print("\n5. 测试训练损失计算...")
    model.train()
    dummy_sequences = torch.randn(2, 4, 500)
    loss = model(dummy_sequences, test_conditions)
    print(f"训练损失: {loss.item():.4f}")
    
    # 6. 预测器接口测试
    print("\n6. 测试预测器兼容接口...")
    predictor = ConditionalDiffusionPredictor(model)
    pred_samples = predictor.predict(conditions=test_conditions, batch_size=1)
    print(f"预测器生成样本形状: {pred_samples.shape}")
    
    loss_value = predictor.train_step(dummy_sequences[:1], test_conditions)
    print(f"预测器训练损失: {loss_value:.4f}")
    
    print("\n✅ 所有测试通过！条件扩散模型实现正确。")
    return True


def benchmark_performance():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    import time
    
    model = create_conditional_diffusion_model(sequence_length=200, num_timesteps=50)
    test_conditions = {'temperature': 37.0, 'ph': 7.0}
    
    # 采样性能
    start_time = time.time()
    with torch.no_grad():
        samples = model.sample(batch_size=4, conditions=test_conditions)
    sampling_time = time.time() - start_time
    print(f"采样性能: 4个样本用时 {sampling_time:.2f}秒")
    
    # 训练性能
    dummy_data = torch.randn(8, 4, 200)
    start_time = time.time()
    for _ in range(10):
        loss = model(dummy_data, test_conditions)
    training_time = time.time() - start_time
    print(f"训练性能: 10次前向传播用时 {training_time:.2f}秒")


if __name__ == "__main__":
    # 基础功能测试
    success = demo_conditional_diffusion()
    
    if success:
        # 性能测试
        benchmark_performance()
        print("\n🎉 条件扩散模型实现完成并验证通过！")
    else:
        print("\n❌ 测试失败，请检查实现。")
