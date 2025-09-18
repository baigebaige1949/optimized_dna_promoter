# 条件扩散模型完整文档

## 概述

条件扩散模型 (Conditional Diffusion Model) 是基于DNA-Diffusion论文实现的条件U-Net架构，支持多种生物学条件的DNA启动子序列生成。

### 主要特性

- ✅ **多条件支持**: 细胞类型、温度、pH、氧气含量、生长周期等
- ✅ **灵活条件控制**: 支持任意条件组合，未提供的条件使用默认值
- ✅ **动态条件**: 支持条件的动态添加和移除
- ✅ **无分类器引导**: 实现Classifier-free Guidance提高生成质量
- ✅ **完全兼容**: 与现有diffusion_model.py保持兼容性

## 架构组件

### 1. ConditionConfig
条件配置类，定义所有可用的条件类型和默认值。

```python
class ConditionConfig:
    # 细胞类型条件
    CELL_TYPES = ['E.coli', 'Yeast', 'HEK293', 'CHO', 'Other']
    
    # 环境条件范围
    TEMPERATURE_RANGE = (25, 42)      # 摄氏度
    PH_RANGE = (5.0, 8.5)            # pH值
    OXYGEN_RANGE = (0, 21)           # 氧气含量百分比
    
    # 生长周期条件
    GROWTH_PHASES = ['Lag', 'Log', 'Stationary', 'Death']
```

### 2. ConditionEmbedding
条件嵌入层，将各种条件编码为统一的嵌入向量。

- **分类条件**: 细胞类型、生长周期使用Embedding层
- **连续条件**: 温度、pH等使用线性投射
- **自动补全**: 未提供的条件使用默认值

### 3. CrossAttention1D
一维交叉注意力机制，用于条件和序列特征的交互。

### 4. ConditionalUNet1D
条件控制的一维U-Net模型，核心架构组件。

- **条件注入**: 在每个残差块中泥入条件信息
- **交叉注意力**: 在关键层使用交叉注意力增强条件控制
- **无分类器引导**: 训练时随机丢弃条件

### 5. ConditionalDiffusionModel
主条件扩散模型，提供完整的条件控制扩散生成功能。

## 使用方法

### 1. 基础使用

```python
from optimized_dna_promoter.core.conditional_diffusion_model import create_conditional_diffusion_model
from optimized_dna_promoter.config.model_config import ConditionalDiffusionModelConfig

# 创建配置
config = ConditionalDiffusionModelConfig()

# 创建模型
model = create_conditional_diffusion_model(config)

# 准备条件
conditions = {
    'cell_type': torch.tensor([0]),      # E.coli
    'temperature': torch.tensor([37.0]), # 37°C
    'ph': torch.tensor([7.0]),          # 中性pH
    'oxygen': torch.tensor([21.0])      # 有氧环境
}

# 生成序列
generated_sequences = model.sample(
    shape=(1, 4, 1000), 
    conditions=conditions,
    guidance_scale=7.5
)
```

### 2. 训练模型

```python
# 准备训练数据
batch = {
    'sequences': torch.randn(16, 4, 1000),  # DNA序列
    'cell_type': torch.tensor([0, 1, 0, 1, ...]),
    'temperature': torch.tensor([37.0, 30.0, ...]),
    'ph': torch.tensor([7.0, 6.5, ...])
}

# 训练步骤
model.train()
result = model.training_step(batch)
loss = result['loss']

# 反向传播
loss.backward()
optimizer.step()
```

### 3. 部分条件生成

```python
# 仅指定部分条件，其余使用默认值
partial_conditions = {
    'cell_type': torch.tensor([1]),      # Yeast
    'temperature': torch.tensor([30.0])  # 低温
    # pH, oxygen等将使用默认值
}

generated = model.sample(
    shape=(1, 4, 500),
    conditions=partial_conditions
)
```

### 4. 无分类器引导控制

```python
# 不同引导强度的效果
guidance_scales = [1.0, 3.0, 7.5, 15.0]

for scale in guidance_scales:
    generated = model.sample(
        shape=(1, 4, 200),
        conditions=conditions,
        guidance_scale=scale  # 调节引导强度
    )
```

## 条件类型详解

### 细胞类型 (cell_type)
- **类型**: 分类变量
- **取值**: 0=E.coli, 1=Yeast, 2=HEK293, 3=CHO, 4=Other
- **默认值**: 0 (E.coli)

### 温度 (temperature)
- **类型**: 连续变量
- **范围**: 25-42°C
- **默认值**: 37.0°C

### pH值 (ph)
- **类型**: 连续变量
- **范围**: 5.0-8.5
- **默认值**: 7.0

### 氧气含量 (oxygen)
- **类型**: 连续变量
- **范围**: 0-21%
- **默认值**: 21.0%

### 生长周期 (growth_phase)
- **类型**: 分类变量
- **取值**: 0=Lag, 1=Log, 2=Stationary, 3=Death
- **默认值**: 1 (Log)

### 营养水平 (nutrient_level)
- **类型**: 连续变量
- **范围**: 0.0-2.0
- **默认值**: 1.0

### 应力水平 (stress_level)
- **类型**: 连续变量
- **范围**: 0.0-1.0
- **默认值**: 0.0

## 配置参数

```python
@dataclass
class ConditionalDiffusionModelConfig(DiffusionModelConfig):
    # 条件相关参数
    condition_embedding_dim: int = 256      # 条件嵌入维度
    use_cross_attention: bool = True        # 使用交叉注意力
    cross_attention_heads: int = 8          # 注意力头数
    
    # 无分类器引导参数
    use_classifier_free_guidance: bool = True
    cfg_drop_prob: float = 0.1             # 条件丢弃概率
    default_guidance_scale: float = 7.5     # 默认引导强度
```

## 高级用法

### 1. 自定义条件验证

```python
from optimized_dna_promoter.core.conditional_diffusion_model import ConditionValidator

validator = ConditionValidator()

# 验证并补全条件
partial_conditions = {'cell_type': torch.tensor([0])}
complete_conditions = validator.validate_and_complete(partial_conditions, batch_size=1)
```

### 2. 批量生成不同条件

```python
# 批量生成多种条件组合
conditions = {
    'cell_type': torch.tensor([0, 1, 2]),       # 3种细胞类型
    'temperature': torch.tensor([30.0, 37.0, 42.0]),
    'ph': torch.tensor([6.5, 7.0, 7.5])
}

generated = model.sample(
    shape=(3, 4, 300),
    conditions=conditions
)
```

### 3. 条件插值生成

```python
# 在两个条件间进行插值
def interpolate_conditions(cond1, cond2, steps=5):
    interpolated = []
    for i in range(steps):
        alpha = i / (steps - 1)
        interp_cond = {}
        for key in cond1:
            if key in ['cell_type', 'growth_phase']:
                # 分类变量不进行插值
                interp_cond[key] = cond1[key] if alpha < 0.5 else cond2[key]
            else:
                # 连续变量进行线性插值
                interp_cond[key] = (1 - alpha) * cond1[key] + alpha * cond2[key]
        interpolated.append(interp_cond)
    return interpolated
```

## 兼容性

### 与基础模型的兼容性

1. **相同的噪声调度器**: 使用相同的NoiseScheduler类
2. **相同的训练接口**: training_step方法保持一致
3. **向下兼容**: 可以不提供条件，作为基础扩散模型使用

### 模型转换

```python
# 从基础模型转换为条件模型
base_config = DiffusionModelConfig()
conditional_config = ConditionalDiffusionModelConfig(
    hidden_dim=base_config.hidden_dim,
    num_timesteps=base_config.num_timesteps,
    # ... 其他相同参数
)

conditional_model = ConditionalDiffusionModel(conditional_config)
```

## 性能优化

1. **梯度检查点**: 支持gradient checkpointing减少显存使用
2. **混合精度**: 支持mixed precision训练
3. **条件缓存**: 条件嵌入可以缓存重用
4. **批量处理**: 支持批量条件生成

## 注意事项

1. **条件范围**: 请确保输入条件在合理范围内
2. **设备兼容**: 模型会自动移动到指定设备
3. **内存使用**: 条件模型参数更多，需要更大显存
4. **训练稳定性**: 建议使用较小的学习率开始训练

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小批次大小
   - 使用gradient checkpointing
   - 减小模型尺寸

2. **条件维度不匹配**
   - 检查条件输入格式
   - 使用ConditionValidator验证

3. **生成质量不佳**
   - 调节guidance_scale
   - 增加训练步数
   - 检查条件设置

### 调试建议

```python
# 启用详细日志
import logging
logging.getLogger('optimized_dna_promoter').setLevel(logging.DEBUG)

# 检查模型结构
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 验证输入形状
for name, tensor in conditions.items():
    print(f"{name}: {tensor.shape} {tensor.dtype}")
```

---

*此文档描述了条件扩散模型的完整功能和使用方法。如有问题，请参考代码注释或联系开发者。*
