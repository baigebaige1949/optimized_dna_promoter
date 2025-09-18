# 版本控制和性能对比系统文档

## 概述

该系统为 DNA 启动子模型提供全面的版本管理、性能对比、实验跟踪和可视化分析功能。通过这个系统，您可以：

- 管理模型的不同版本
- 比较模型性能并生成详细报告
- 跟踪实验过程和结果
- 创建丰富的可视化图表
- 实现 A/B 测试和模型回滚

## 核心组件

### 1. ModelVersionManager - 模型版本管理器

负责模型版本的保存、加载和管理。

#### 主要功能：
- `save_version()`: 保存模型版本
- `load_version()`: 加载模型版本
- `list_versions()`: 列出所有版本
- `delete_version()`: 删除指定版本
- `compare_versions()`: 比较两个版本
- `create_checkpoint()`: 创建训练检查点

#### 使用示例：
```python
from version_control import ModelVersionManager

# 初始化
version_manager = ModelVersionManager("./model_versions")

# 保存模型版本
version_name = version_manager.save_version(
    model=your_model,
    version_name="dna_predictor_v1",
    description="第一个版本的DNA预测模型",
    metadata={"accuracy": 0.95, "epoch": 100}
)

# 加载模型版本
loaded_model = version_manager.load_model_weights("dna_predictor_v1", target_model)

# 列出所有版本
versions = version_manager.list_versions()
for version in versions:
    print(f"Version: {version['version_name']}, Created: {version['created_at']}")
```

### 2. PerformanceComparator - 性能对比分析器

用于评估和比较多个模型的性能。

#### 主要功能：
- `evaluate_model()`: 评估单个模型性能
- `compare_models()`: 比较多个模型
- `generate_comparison_report()`: 生成比较报告
- `ab_test()`: 进行 A/B 测试

#### 使用示例：
```python
from version_control import PerformanceComparator

# 初始化
comparator = PerformanceComparator("./comparison_results")

# 评估模型
result1 = comparator.evaluate_model(
    model=model1, 
    data_loader=test_loader, 
    criterion=nn.CrossEntropyLoss(),
    model_name="Model_A"
)

result2 = comparator.evaluate_model(
    model=model2, 
    data_loader=test_loader, 
    criterion=nn.CrossEntropyLoss(),
    model_name="Model_B"
)

# 比较模型
comparison = comparator.compare_models([result1, result2], "AB_Comparison")

# 生成报告
report_path = comparator.generate_comparison_report(comparison)

# A/B 测试
ab_result = comparator.ab_test(result1, result2)
print(f"Recommended model: {ab_result['summary']['recommendation']}")
```

### 3. ExperimentTracker - 实验跟踪器

负责记录和管理实验过程及结果。

#### 主要功能：
- `start_experiment()`: 开始新实验
- `log_hyperparameters()`: 记录超参数
- `log_metrics()`: 记录评估指标
- `log_artifact()`: 记录实验产物
- `end_experiment()`: 结束实验
- `compare_experiments()`: 比较多个实验

#### 使用示例：
```python
from version_control import ExperimentTracker

# 初始化
tracker = ExperimentTracker("./experiments")

# 开始实验
experiment_id = tracker.start_experiment(
    "DNA_Prediction_Experiment",
    "DNA序列启动子预测模型训练",
    tags=["dna", "prediction", "transformer"]
)

# 记录超参数
tracker.log_hyperparameters(experiment_id, {
    "learning_rate": 0.001,
    "batch_size": 32,
    "hidden_size": 256
})

# 训练过程中记录指标
for epoch in range(epochs):
    # ... 训练代码 ...
    tracker.log_metrics(experiment_id, {
        "train_loss": train_loss,
        "val_accuracy": val_acc
    }, epoch=epoch)

# 结束实验
tracker.end_experiment(experiment_id, "completed")
```

### 4. VisualizationManager - 可视化管理器

生成各种图表和可视化结果。

#### 主要功能：
- `plot_training_curves()`: 绘制训练曲线
- `plot_metric_comparison()`: 绘制指标对比图
- `plot_performance_radar()`: 绘制性能雷达图
- `plot_model_complexity_comparison()`: 绘制模型复杂度对比
- `create_dashboard()`: 创建综合仪表盘

#### 使用示例：
```python
from version_control import VisualizationManager

# 初始化
viz_manager = VisualizationManager("./visualizations")

# 绘制训练曲线
experiment_data = tracker.get_experiment(experiment_id)
curve_path = viz_manager.plot_training_curves(experiment_data)

# 绘制性能对比
comparison_path = viz_manager.plot_metric_comparison(
    [comparison_result], "accuracy"
)

# 生成仪表盘
dashboard_path = viz_manager.create_dashboard(
    experiment_data, [comparison_result]
)
```

## 集成使用

### EnhancedTrainer - 增强训练器

`EnhancedTrainer` 将所有功能集成在一起，提供统一的接口。

```python
from version_control.integration import EnhancedTrainer

# 初始化
trainer = EnhancedTrainer(
    base_dir="./training_workspace",
    auto_versioning=True,
    auto_checkpoint=True,
    checkpoint_interval=10
)

# 开始实验
trainer.start_experiment(
    "DNA_Model_Training",
    "DNA序列预测模型训练实验",
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
)

# 训练模型时自动记录版本和指标
result = trainer.train_model(
    model, train_loader, val_loader, 
    criterion, optimizer, epochs=100
)

# 结束实验
trainer.end_experiment()

# 比较多个模型
comparison = trainer.evaluate_and_compare_models([
    (model1, test_loader, "Model_A"),
    (model2, test_loader, "Model_B")
])

# 模型回滚
rollback_model = trainer.rollback_to_version("best_version", new_model)
```

## 配置选项

### 目录结构
```
training_workspace/
├── versions/          # 模型版本存储
│   ├── model_v1/
│   └── checkpoints/
├── experiments/       # 实验数据
│   ├── experiment_1/
│   └── experiments.json
├── results/           # 性能对比结果
│   ├── comparison_1.json
│   └── reports/
└── visualizations/    # 可视化结果
    ├── training_curves.png
    └── comparison_charts.png
```

### 可配置参数

- `auto_versioning`: 是否自动保存模型版本
- `auto_checkpoint`: 是否自动保存检查点
- `checkpoint_interval`: 检查点保存间隔轮数
- `max_versions`: 最大保留版本数

## 最佳实践

### 1. 版本命名规范
- 使用语义化版本号：`model_v1.0.0`
- 包含日期信息：`dna_predictor_20231201`
- 标记特殊版本：`baseline_model`, `best_accuracy`

### 2. 实验管理
- 每个实验都应该有清晰的描述和目标
- 记录所有相关的超参数和配置
- 定期保存检查点以防止数据丢失

### 3. 性能对比
- 在相同的数据集上进行公平比较
- 考虑多个指标，不仅仅是准确率
- 记录推理时间和内存使用情况

### 4. 模型部署
- 在部署前进行充分的 A/B 测试
- 保留上一个稳定版本以便回滚
- 监控生产环境中的模型性能

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型结构是否一致
   - 确认版本文件存在且没有损坏

2. **实验数据丢失**
   - 检查磁盘空间是否充足
   - 确认文件权限设置正确

3. **可视化生成失败**
   - 检查 matplotlib 和 seaborn 安装
   - 确保数据格式正确

### 日志检查
所有操作都会记录在实验日志中，可以通过以下方式查看：

```python
# 查看实验日志
experiment_data = tracker.get_experiment(experiment_id)
for log_entry in experiment_data['logs']:
    print(f"{log_entry['timestamp']}: {log_entry['message']}")
```

## API 参考

### ModelVersionManager API

#### save_version(model, version_name, metadata=None, description="", auto_increment=False)
保存模型版本。

**参数：**
- `model`: PyTorch 模型
- `version_name`: 版本名称
- `metadata`: 元数据字典
- `description`: 版本描述
- `auto_increment`: 自动递增版本号

**返回：** 实际保存的版本名称

#### load_version(version_name)
加载模型版本信息。

**参数：**
- `version_name`: 版本名称

**返回：** 包含模型信息的字典

### PerformanceComparator API

#### evaluate_model(model, data_loader, criterion, device='cpu', model_name="model")
评估模型性能。

**参数：**
- `model`: PyTorch 模型
- `data_loader`: 数据加载器
- `criterion`: 损失函数
- `device`: 计算设备
- `model_name`: 模型名称

**返回：** 评估结果字典

#### compare_models(model_results, comparison_name=None)
比较多个模型性能。

**参数：**
- `model_results`: 模型评估结果列表
- `comparison_name`: 比较名称

**返回：** 比较结果字典

更多 API 详情请参考源代码注释。

## 总结

这个版本控制和性能对比系统为 DNA 启动子模型的研发和部署提供了全面的支持。通过统一的接口和丰富的功能，可以大大提高模型开发的效率和质量。

建议在使用过程中：
1. 遵循最佳实践指南
2. 定期备份重要数据
3. 监控系统资源使用情况
4. 及时清理旧的版本和实验数据

如有问题或建议，请查看故障排除部分或联系开发团队。
