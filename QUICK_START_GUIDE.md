# DNA启动子优化系统 - 快速入门指南

## 概述

本指南将帮助您在5分钟内快速上手DNA启动子优化系统，包括条件控制生成、GPU训练和结果分析的完整流程。

## 一、环境准备

### 1.1 系统要求
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+（推荐使用GPU）
- 至少8GB RAM（GPU训练推荐16GB+）

### 1.2 快速安装
```bash
# 克隆项目
git clone <your-repo-url>
cd optimized_dna_promoter

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 1.3 验证安装
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
```

## 二、30秒快速体验

### 2.1 一键运行示例
```bash
python one_click_train.py --quick-demo
```

这将：
- 使用内置示例数据
- 训练一个小型模型（5分钟）
- 生成条件控制的DNA序列
- 输出性能报告

### 2.2 查看结果
```bash
# 查看生成的序列
cat results/generated_sequences.fasta

# 查看训练报告
cat results/training_report.json
```

## 三、完整使用流程

### 3.1 数据准备

#### 支持的数据格式
```python
from optimized_dna_promoter.data import create_enhanced_dataset

# 创建数据集对象
dataset = create_enhanced_dataset(
    max_length=200,
    vocab_size=4,
    enable_augmentation=True
)

# 支持多种格式加载
dataset.load_from_file('data/sequences.fasta')        # FASTA格式
dataset.load_from_file('data/promoters.csv')          # CSV格式
dataset.load_from_file('data/genbank_data.gb')       # GenBank格式
```

#### 数据质量检查
```python
# 自动数据质量检查
quality_report = dataset.quality_check()
print("数据质量报告:")
print(f"总序列数: {quality_report['total_sequences']}")
print(f"平均长度: {quality_report['avg_length']:.1f}")
print(f"GC含量范围: {quality_report['gc_range']}")

# 数据清洗
dataset.clean_data(
    min_length=50,
    max_length=500,
    min_gc=0.3,
    max_gc=0.7
)
```

### 3.2 条件控制设置

#### 创建条件控制系统
```python
from optimized_dna_promoter.conditions import create_condition_system

# 创建条件控制器和智能填充器
controller, filler = create_condition_system()

# 定义目标条件（可以只设置部分条件）
target_conditions = {
    'temperature': 37.0,     # 温度（摄氏度）
    'ph': 7.4,              # pH值
    'cell_type': 'E.coli',   # 细胞类型
    'oxygen_level': 21.0,    # 氧气含量（%）
    'growth_phase': 'log'    # 生长期
}

# 创建条件向量
condition_vector = controller.create_condition_vector(target_conditions)
```

#### 智能条件填充
```python
# 只设置部分关键条件，让系统智能填充其余条件
partial_conditions = {
    'temperature': 37.0,
    'cell_type': 'E.coli'
}

# 基于生物学知识智能填充
filled_conditions = filler.intelligent_fill(
    partial_conditions,
    biological_context='prokaryotic',
    target_pathways=['glycolysis', 'central_metabolism']
)

print("智能填充后的完整条件:")
for key, value in filled_conditions.conditions.items():
    print(f"{key}: {value}")
```

### 3.3 模型训练

#### 快速训练（适合快速验证）
```python
from optimized_dna_promoter.training import create_advanced_trainer
from optimized_dna_promoter.models import create_multimodal_fusion_model

# 创建模型
model = create_multimodal_fusion_model(
    vocab_size=5,
    seq_len=200,
    embed_dim=256,
    hidden_dim=512,
    output_dim=1
)

# 快速训练配置
trainer = create_advanced_trainer(
    model=model,
    config={
        'learning_rate': 1e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'early_stopping_patience': 10,
        'use_amp': True,  # 自动混合精度
        'gradient_accumulation_steps': 4
    }
)

# 开始训练
training_results = trainer.train(
    train_dataset=dataset.get_train_dataset(),
    val_dataset=dataset.get_val_dataset()
)
```

#### GPU最佳实践设置
```python
# GPU优化配置
gpu_config = {
    # 内存优化
    'batch_size': 64 if torch.cuda.get_device_properties(0).total_memory > 8e9 else 32,
    'num_workers': min(8, torch.multiprocessing.cpu_count()),
    'pin_memory': True,
    'non_blocking': True,
    
    # 训练优化
    'use_amp': True,                    # 自动混合精度
    'gradient_checkpointing': True,     # 梯度检查点
    'compile_model': True,              # PyTorch 2.0编译
    
    # 内存管理
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 4,
    'empty_cache_steps': 100
}

trainer.update_config(gpu_config)
```

### 3.4 条件控制生成

#### 基础生成
```python
from optimized_dna_promoter.generation import create_generation_pipeline

# 创建生成流水线
pipeline = create_generation_pipeline({
    'noise_scheduler': 'cosine',
    'sampler': 'dpm_solver_plus',
    'post_process': True,
    'biological_constraints': True
})

# 条件控制生成
generated_sequences = pipeline.generate(
    model=model,
    conditions=filled_conditions.to_tensor(),
    batch_size=16,
    seq_length=200,
    num_steps=50
)

print(f"成功生成 {len(generated_sequences)} 个DNA序列")
```

#### 高级自适应生成
```python
# 自适应生成 - 根据目标条件优化生成质量
best_sequences = pipeline.adaptive_generation(
    model=model,
    target_conditions={
        'temperature': 37.0,
        'expected_strength': 0.8,  # 期望的启动子强度
        'gc_content_range': (0.4, 0.6)
    },
    batch_size=32,
    max_iterations=10,
    quality_threshold=0.85
)
```

### 3.5 结果分析和评估

#### 生物学指标评估
```python
from optimized_dna_promoter.evaluation import BiologicalMetrics

evaluator = BiologicalMetrics()

# 全面评估生成的序列
eval_results = evaluator.evaluate_batch(
    sequences=generated_sequences,
    reference_conditions=filled_conditions.conditions,
    metrics=['gc_content', 'complexity', 'motif_analysis', 'secondary_structure']
)

print("生物学评估结果:")
for metric, value in eval_results.items():
    print(f"{metric}: {value:.3f}")
```

#### 条件符合度分析
```python
# 检查生成序列是否符合指定条件
compliance_report = evaluator.check_condition_compliance(
    sequences=generated_sequences,
    target_conditions=filled_conditions.conditions
)

print("条件符合度报告:")
for condition, compliance in compliance_report.items():
    print(f"{condition}: {compliance*100:.1f}% 符合度")
```

## 四、一键式训练脚本

### 4.1 基本用法
```bash
# 使用默认配置训练
python one_click_train.py --data-path data/promoters.csv

# 指定条件进行训练
python one_click_train.py \
    --data-path data/promoters.csv \
    --conditions temperature=37,ph=7.4,cell_type=ecoli \
    --epochs 100 \
    --batch-size 64
```

### 4.2 高级参数
```bash
# GPU优化训练
python one_click_train.py \
    --data-path data/promoters.csv \
    --gpu-optimization \
    --mixed-precision \
    --distributed \
    --gradient-checkpointing

# 超参数自动调优
python one_click_train.py \
    --data-path data/promoters.csv \
    --auto-tune \
    --n-trials 50
```

### 4.3 输出文件说明
训练完成后将在 `results/` 目录生成：
- `model_best.pth` - 最佳模型权重
- `training_log.json` - 详细训练日志
- `generated_sequences.fasta` - 生成的DNA序列
- `evaluation_report.json` - 评估报告
- `condition_analysis.pdf` - 条件分析图表

## 五、常见问题解答

### Q1: 内存不足怎么办？
```python
# 减少批量大小
config['batch_size'] = 16

# 启用梯度累积
config['gradient_accumulation_steps'] = 8

# 启用梯度检查点
config['gradient_checkpointing'] = True
```

### Q2: 如何加速训练？
```python
# 启用混合精度
config['use_amp'] = True

# 使用编译模式（PyTorch 2.0+）
model = torch.compile(model)

# 增加数据加载器工作进程
config['num_workers'] = 8
```

### Q3: 如何自定义条件类型？
```python
# 扩展条件控制器
controller.add_custom_condition(
    name='custom_stress',
    value_range=(0.0, 10.0),
    default_value=1.0,
    data_type='continuous'
)

# 使用自定义条件
custom_conditions = {
    'temperature': 42.0,
    'custom_stress': 5.0
}
```

### Q4: 生成质量不理想怎么办？
```python
# 调整生成参数
pipeline_config = {
    'num_steps': 100,           # 增加采样步数
    'guidance_scale': 7.5,      # 调整引导强度
    'post_process': True,       # 启用后处理
    'quality_filter': True      # 质量过滤
}

# 使用自适应生成
best_sequences = pipeline.adaptive_generation(
    quality_threshold=0.9,      # 提高质量阈值
    max_iterations=20          # 增加迭代次数
)
```

## 六、进阶使用技巧

### 6.1 批量条件生成
```python
# 生成多种条件组合
condition_combinations = controller.generate_condition_combinations(
    temperature_range=(30, 42),
    ph_range=(6.5, 8.0),
    cell_types=['E.coli', 'B.subtilis'],
    n_combinations=50
)

# 批量生成
all_sequences = []
for conditions in condition_combinations:
    sequences = pipeline.generate(
        model=model,
        conditions=conditions.to_tensor(),
        batch_size=8
    )
    all_sequences.extend(sequences)
```

### 6.2 实时监控训练
```python
# 启用TensorBoard监控
trainer.enable_tensorboard(log_dir='tensorboard_logs')

# 启用Weights & Biases
trainer.enable_wandb(project_name='dna_promoter_optimization')

# 自定义回调函数
def custom_callback(trainer, epoch, metrics):
    if metrics['val_loss'] < 0.1:
        print(f"达到目标损失，第{epoch}轮训练")
        trainer.early_stop()

trainer.add_callback('epoch_end', custom_callback)
```

### 6.3 模型版本管理
```python
from optimized_dna_promoter.version_control import ModelVersionManager

# 版本管理
version_manager = ModelVersionManager('models/')

# 保存带标签的模型版本
version_manager.save_model(
    model=model,
    version_tag='v1.2.0',
    metadata={
        'training_data': 'ecoli_promoters_v3.csv',
        'conditions': str(target_conditions),
        'performance': training_results['best_val_acc']
    }
)

# 加载特定版本
model_v1 = version_manager.load_model('v1.2.0')
```

## 七、技术支持

如果您在使用过程中遇到问题：

1. 查看 `docs/` 目录下的详细文档
2. 检查 `examples/` 目录下的示例代码
3. 查看 GitHub Issues 或提交新问题
4. 参考性能分析报告 `PERFORMANCE_ANALYSIS.md`

---

**恭喜！** 您已经掌握了DNA启动子优化系统的基本使用方法。现在可以开始您的DNA序列设计和优化工作了！