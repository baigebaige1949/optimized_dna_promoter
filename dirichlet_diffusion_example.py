"""
Dirichlet扩散模型使用示例
演示如何使用新的DDSM模型进行DNA序列生成和训练
"""

import torch
import torch.nn.functional as F
from core.dirichlet_diffusion import (
    DirichletDiffusionModel,
    DDSMInterface,
    StickBreakingTransform
)

def create_sample_dna_data(batch_size: int = 32, seq_length: int = 100):
    """
    创建示例DNA数据用于测试
    
    Returns:
        one_hot_sequences: (batch_size, seq_length, 4) 的one-hot编码DNA序列
    """
    # 随机生成DNA序列 (A=0, T=1, G=2, C=3)
    categorical_sequences = torch.randint(0, 4, (batch_size, seq_length))
    
    # 转换为one-hot编码
    one_hot_sequences = F.one_hot(categorical_sequences, num_classes=4).float()
    
    return one_hot_sequences

def basic_usage_example():
    """
    基本使用示例：模型初始化、训练和采样
    """
    print("=== Dirichlet扩散模型基本使用示例 ===")
    
    # 1. 创建模型
    model = DirichletDiffusionModel(
        sequence_length=100,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        alpha=2.0,
        beta=2.0,
        dilation_factor=2.0
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 创建示例数据
    batch_size = 16
    seq_length = 100
    sample_data = create_sample_dna_data(batch_size, seq_length)
    
    print(f"样本数据形状: {sample_data.shape}")
    
    # 3. 计算训练损失
    model.train()
    losses = model.compute_loss(sample_data)
    
    print(f"训练损失: {losses['total_loss'].item():.4f}")
    
    # 4. 生成样本
    model.eval()
    with torch.no_grad():
        generated_samples = model.sample(
            batch_size=8,
            sequence_length=seq_length,
            num_steps=50,
            temperature=1.0
        )
    
    print(f"生成样本形状: {generated_samples.shape}")
    print(f"生成样本概率和: {generated_samples.sum(dim=-1).mean():.4f} (应该接近1.0)")
    
    # 5. 计算似然
    with torch.no_grad():
        log_likelihood = model.compute_likelihood(sample_data[:4], num_steps=20)
    
    print(f"样本对数似然: {log_likelihood.mean().item():.4f}")

def stick_breaking_example():
    """
    Stick-breaking变换示例
    """
    print("\n=== Stick-breaking变换示例 ===")
    
    transform = StickBreakingTransform()
    
    # 创建示例概率分布
    batch_size = 5
    seq_length = 10
    
    # 创建归一化的概率分布
    probs = torch.rand(batch_size, seq_length, 4)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    print(f"原始概率分布形状: {probs.shape}")
    print(f"概率和验证: {probs.sum(dim=-1)[:2]}")  # 打印前两个样本的概率和
    
    # 转换为stick-breaking参数
    stick_breaking_params = transform.simplex_to_stick_breaking(probs)
    print(f"Stick-breaking参数形状: {stick_breaking_params.shape}")
    
    # 转换回概率分布
    reconstructed_probs = transform.stick_breaking_to_simplex(stick_breaking_params)
    print(f"重构概率分布形状: {reconstructed_probs.shape}")
    
    # 检查重构误差
    reconstruction_error = torch.mean(torch.abs(probs - reconstructed_probs))
    print(f"重构误差: {reconstruction_error.item():.6f}")

def interface_conversion_example():
    """
    接口转换示例：与现有扩散模型的兼容性
    """
    print("\n=== 接口转换示例 ===")
    
    # 创建模型和接口
    model = DirichletDiffusionModel(sequence_length=50, hidden_dim=128)
    interface = DDSMInterface(model)
    
    batch_size = 8
    seq_length = 50
    
    # 1. 从one-hot格式转换
    one_hot_data = create_sample_dna_data(batch_size, seq_length)
    stick_breaking_data = interface.convert_from_standard_diffusion(
        one_hot_data, format_type="one_hot"
    )
    
    print(f"One-hot -> Stick-breaking: {one_hot_data.shape} -> {stick_breaking_data.shape}")
    
    # 2. 从类别格式转换
    categorical_data = torch.randint(0, 4, (batch_size, seq_length))
    stick_breaking_from_cat = interface.convert_from_standard_diffusion(
        categorical_data, format_type="categorical"
    )
    
    print(f"Categorical -> Stick-breaking: {categorical_data.shape} -> {stick_breaking_from_cat.shape}")
    
    # 3. 转换回标准格式
    reconstructed_one_hot = interface.convert_to_standard_diffusion(
        stick_breaking_data, format_type="one_hot"
    )
    reconstructed_categorical = interface.convert_to_standard_diffusion(
        stick_breaking_data, format_type="categorical"
    )
    
    print(f"Stick-breaking -> One-hot: {stick_breaking_data.shape} -> {reconstructed_one_hot.shape}")
    print(f"Stick-breaking -> Categorical: {stick_breaking_data.shape} -> {reconstructed_categorical.shape}")
    
    # 4. 包装训练步骤
    losses = interface.wrap_training_step(one_hot_data)
    print(f"包装训练损失: {losses['total_loss'].item():.4f}")
    
    # 5. 包装采样
    samples = interface.wrap_sampling(
        batch_size=4,
        sequence_length=seq_length,
        num_steps=30,
        temperature=0.8
    )
    print(f"包装采样结果: {samples.shape}")

def advanced_features_example():
    """
    高级功能示例：时间膨胀、重要性采样等
    """
    print("\n=== 高级功能示例 ===")
    
    from core.dirichlet_diffusion import TimeDilation, VariationalScoreMatching, JacobiProcess
    
    # 1. 时间膨胀示例
    time_dilation = TimeDilation(dilation_factor=3.0)
    
    original_times = torch.linspace(0, 1, 10)
    dilated_times = time_dilation.forward_time(original_times)
    reconstructed_times = time_dilation.inverse_time(dilated_times)
    
    print("时间膨胀效果:")
    print(f"原始时间: {original_times[:5]}")
    print(f"膨胀时间: {dilated_times[:5]}")
    print(f"重构时间: {reconstructed_times[:5]}")
    
    # 2. Jacobi过程示例
    jacobi = JacobiProcess(alpha=2.0, beta=2.0)
    
    # 从先验分布采样
    prior_samples = jacobi.sample_prior((5, 10, 3), device='cpu')
    print(f"Jacobi先验样本形状: {prior_samples.shape}")
    print(f"先验样本范围: [{prior_samples.min().item():.3f}, {prior_samples.max().item():.3f}]")
    
    # 计算漂移和扩散
    t_samples = torch.rand(5, 1)
    drift = jacobi.drift(prior_samples, t_samples)
    diffusion = jacobi.diffusion(prior_samples, t_samples)
    
    print(f"漂移项形状: {drift.shape}")
    print(f"扩散项形状: {diffusion.shape}")
    
    # 3. 重要性采样权重
    score_matching = VariationalScoreMatching()
    importance_weights = score_matching.compute_importance_weights(
        prior_samples, t_samples, jacobi
    )
    
    print(f"重要性权重形状: {importance_weights.shape}")
    print(f"权重统计: 均值={importance_weights.mean().item():.4f}, 标准差={importance_weights.std().item():.4f}")

def dna_sequence_quality_analysis():
    """
    DNA序列生成质量分析
    """
    print("\n=== DNA序列生成质量分析 ===")
    
    model = DirichletDiffusionModel(
        sequence_length=200,
        hidden_dim=256,
        alpha=1.5,  # 调整参数以获得更自然的序列
        beta=1.5
    )
    
    # 生成多个样本进行分析
    num_samples = 20
    seq_length = 200
    
    model.eval()
    with torch.no_grad():
        generated_samples = model.sample(
            batch_size=num_samples,
            sequence_length=seq_length,
            num_steps=100,
            temperature=0.9
        )
    
    # 转换为类别表示进行分析
    categorical_sequences = torch.argmax(generated_samples, dim=-1)
    
    # 1. 碱基组成分析
    base_counts = torch.bincount(categorical_sequences.flatten(), minlength=4)
    base_frequencies = base_counts.float() / base_counts.sum()
    base_names = ['A', 'T', 'G', 'C']
    
    print("碱基组成分析:")
    for i, (base, freq) in enumerate(zip(base_names, base_frequencies)):
        print(f"  {base}: {freq.item():.3f} ({base_counts[i].item()}个)")
    
    # 2. GC含量分析
    gc_content = (base_frequencies[2] + base_frequencies[3]).item()  # G + C
    print(f"GC含量: {gc_content:.3f}")
    
    # 3. 序列多样性分析
    unique_sequences = len(set(tuple(seq.tolist()) for seq in categorical_sequences))
    print(f"唯一序列数量: {unique_sequences}/{num_samples} (多样性: {unique_sequences/num_samples:.3f})")
    
    # 4. 重复模式分析
    def count_repeats(sequences, repeat_length=3):
        repeat_counts = {}
        for seq in sequences:
            for i in range(len(seq) - repeat_length + 1):
                pattern = tuple(seq[i:i+repeat_length].tolist())
                repeat_counts[pattern] = repeat_counts.get(pattern, 0) + 1
        return repeat_counts
    
    repeats = count_repeats(categorical_sequences, repeat_length=3)
    top_repeats = sorted(repeats.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("最常见的3-mer模式:")
    for pattern, count in top_repeats:
        pattern_str = ''.join(base_names[i] for i in pattern)
        print(f"  {pattern_str}: {count}次")

if __name__ == "__main__":
    # 设置随机种子以保证可重现性
    torch.manual_seed(42)
    
    try:
        # 运行所有示例
        basic_usage_example()
        stick_breaking_example()
        interface_conversion_example()
        advanced_features_example()
        dna_sequence_quality_analysis()
        
        print("\n=== 所有示例运行完成！ ===")
        print("\n使用建议:")
        print("1. 对于短序列(<100bp)，建议使用较小的dilation_factor(1.5-2.0)")
        print("2. 对于长序列(>500bp)，建议增加采样步数到200+")
        print("3. alpha和beta参数控制Jacobi过程的特性，建议从1.5-3.0范围内调整")
        print("4. 使用importance sampling可以显著减少训练方差")
        print("5. 生成时降低temperature(0.7-0.9)可以获得更稳定的序列")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        print("请检查依赖包是否正确安装，模型实现是否有误")
