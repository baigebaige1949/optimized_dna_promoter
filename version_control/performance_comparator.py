"""性能对比分析器

实现模型性能对比、评估报告生成等功能
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PerformanceComparator:
    """性能对比分析器"""
    
    def __init__(self, results_dir: str = "./comparison_results"):
        """初始化性能对比器
        
        Args:
            results_dir: 结果存储目录
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def evaluate_model(self, 
                      model: nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      criterion: nn.Module,
                      device: str = 'cpu',
                      model_name: str = "model") -> Dict[str, Any]:
        """评估单个模型性能
        
        Args:
            model: 要评估的模型
            data_loader: 数据加载器
            criterion: 损失函数
            device: 计算设备
            model_name: 模型名称
            
        Returns:
            评估结果
        """
        model.eval()
        model = model.to(device)
        
        total_loss = 0.0
        predictions = []
        targets = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                
                # 测量推理时间
                start_time = datetime.now()
                output = model(data)
                end_time = datetime.now()
                
                inference_time = (end_time - start_time).total_seconds()
                inference_times.append(inference_time)
                
                # 计算损失
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # 收集预测结果
                if output.dim() > 1 and output.size(1) > 1:
                    pred = output.argmax(dim=1)
                else:
                    pred = (output > 0.5).float()
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # 计算各项指标
        avg_loss = total_loss / len(data_loader)
        avg_inference_time = np.mean(inference_times)
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 根据任务类型计算指标
        if len(np.unique(targets)) > 2:  # 多分类
            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions, average='weighted')
            recall = recall_score(targets, predictions, average='weighted')
            f1 = f1_score(targets, predictions, average='weighted')
        else:  # 二分类或回归
            if np.all(np.isin(targets, [0, 1])):  # 二分类
                accuracy = accuracy_score(targets, predictions)
                precision = precision_score(targets, predictions)
                recall = recall_score(targets, predictions)
                f1 = f1_score(targets, predictions)
            else:  # 回归任务
                accuracy = None
                precision = None
                recall = None
                f1 = None
        
        # 计算模型复杂度指标
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        
        results = {
            'model_name': model_name,
            'evaluation_time': datetime.now().isoformat(),
            'metrics': {
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'performance': {
                'avg_inference_time': avg_inference_time,
                'total_inference_time': sum(inference_times),
                'samples_per_second': len(predictions) / sum(inference_times) if sum(inference_times) > 0 else 0
            },
            'model_complexity': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb
            },
            'data_info': {
                'num_samples': len(predictions),
                'num_batches': len(data_loader)
            }
        }
        
        return results
    
    def compare_models(self, 
                      model_results: List[Dict[str, Any]],
                      comparison_name: str = None) -> Dict[str, Any]:
        """比较多个模型性能
        
        Args:
            model_results: 模型评估结果列表
            comparison_name: 比较名称
            
        Returns:
            比较结果
        """
        if comparison_name is None:
            comparison_name = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 提取关键指标
        models_data = []
        for result in model_results:
            model_data = {
                'name': result['model_name'],
                'loss': result['metrics']['loss'],
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1_score': result['metrics']['f1_score'],
                'inference_time': result['performance']['avg_inference_time'],
                'samples_per_second': result['performance']['samples_per_second'],
                'parameters': result['model_complexity']['total_parameters'],
                'model_size_mb': result['model_complexity']['model_size_mb']
            }
            models_data.append(model_data)
        
        # 创建DataFrame进行分析
        df = pd.DataFrame(models_data)
        
        # 计算排名
        rankings = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'samples_per_second']:
            if df[metric].notna().any():
                rankings[f'{metric}_rank'] = df[metric].rank(ascending=False, method='min')
        
        for metric in ['loss', 'inference_time', 'parameters', 'model_size_mb']:
            if df[metric].notna().any():
                rankings[f'{metric}_rank'] = df[metric].rank(ascending=True, method='min')
        
        # 添加排名到DataFrame
        for rank_name, rank_series in rankings.items():
            df[rank_name] = rank_series
        
        # 计算综合评分
        score_weights = {
            'accuracy': 0.25,
            'f1_score': 0.25,
            'samples_per_second': 0.2,
            'model_size_mb': -0.15,  # 负权重，模型越小越好
            'inference_time': -0.15  # 负权重，时间越短越好
        }
        
        composite_scores = []
        for _, row in df.iterrows():
            score = 0
            for metric, weight in score_weights.items():
                if pd.notna(row[metric]):
                    # 归一化到0-1范围
                    if weight > 0:
                        normalized = (row[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                    else:
                        normalized = (df[metric].max() - row[metric]) / (df[metric].max() - df[metric].min())
                    score += abs(weight) * normalized
            composite_scores.append(score)
        
        df['composite_score'] = composite_scores
        df['overall_rank'] = df['composite_score'].rank(ascending=False, method='min')
        
        # 生成比较报告
        comparison_result = {
            'comparison_name': comparison_name,
            'comparison_time': datetime.now().isoformat(),
            'models_count': len(model_results),
            'detailed_results': model_results,
            'summary_table': df.to_dict('records'),
            'best_models': {
                'highest_accuracy': df.loc[df['accuracy'].idxmax()]['name'] if df['accuracy'].notna().any() else None,
                'fastest_inference': df.loc[df['inference_time'].idxmin()]['name'] if df['inference_time'].notna().any() else None,
                'smallest_model': df.loc[df['model_size_mb'].idxmin()]['name'] if df['model_size_mb'].notna().any() else None,
                'overall_best': df.loc[df['overall_rank'].idxmin()]['name']
            },
            'statistics': {
                'accuracy': df['accuracy'].describe().to_dict() if df['accuracy'].notna().any() else None,
                'inference_time': df['inference_time'].describe().to_dict() if df['inference_time'].notna().any() else None,
                'model_size': df['model_size_mb'].describe().to_dict() if df['model_size_mb'].notna().any() else None
            }
        }
        
        # 保存比较结果
        results_file = self.results_dir / f"{comparison_name}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Comparison results saved to {results_file}")
        return comparison_result
    
    def generate_comparison_report(self, 
                                  comparison_result: Dict[str, Any],
                                  output_file: str = None) -> str:
        """生成比较报告
        
        Args:
            comparison_result: 比较结果
            output_file: 输出文件路径
            
        Returns:
            报告文件路径
        """
        if output_file is None:
            output_file = self.results_dir / f"{comparison_result['comparison_name']}_report.md"
        else:
            output_file = Path(output_file)
        
        report_content = f"""# 模型性能对比报告

## 基本信息
- **比较名称**: {comparison_result['comparison_name']}
- **比较时间**: {comparison_result['comparison_time']}
- **模型数量**: {comparison_result['models_count']}

## 最佳模型
- **准确率最高**: {comparison_result['best_models']['highest_accuracy']}
- **推理最快**: {comparison_result['best_models']['fastest_inference']}
- **模型最小**: {comparison_result['best_models']['smallest_model']}
- **综合最佳**: {comparison_result['best_models']['overall_best']}

## 详细对比结果

| 模型名称 | 准确率 | 精确率 | 召回率 | F1分数 | 推理时间(s) | 模型大小(MB) | 综合排名 |
|---------|-------|-------|-------|-------|-----------|------------|----------|
"""
        
        for model in comparison_result['summary_table']:
            report_content += f"| {model['name']} | {model['accuracy']:.4f if model['accuracy'] else 'N/A'} | {model['precision']:.4f if model['precision'] else 'N/A'} | {model['recall']:.4f if model['recall'] else 'N/A'} | {model['f1_score']:.4f if model['f1_score'] else 'N/A'} | {model['inference_time']:.6f if model['inference_time'] else 'N/A'} | {model['model_size_mb']:.2f if model['model_size_mb'] else 'N/A'} | {int(model['overall_rank'])} |\n"
        
        # 添加统计信息
        if comparison_result['statistics']['accuracy']:
            report_content += f"""

## 统计信息

### 准确率统计
- 均值: {comparison_result['statistics']['accuracy']['mean']:.4f}
- 标准差: {comparison_result['statistics']['accuracy']['std']:.4f}
- 最小值: {comparison_result['statistics']['accuracy']['min']:.4f}
- 最大值: {comparison_result['statistics']['accuracy']['max']:.4f}
"""
        
        if comparison_result['statistics']['inference_time']:
            report_content += f"""
### 推理时间统计
- 均值: {comparison_result['statistics']['inference_time']['mean']:.6f}s
- 标准差: {comparison_result['statistics']['inference_time']['std']:.6f}s
- 最小值: {comparison_result['statistics']['inference_time']['min']:.6f}s
- 最大值: {comparison_result['statistics']['inference_time']['max']:.6f}s
"""
        
        if comparison_result['statistics']['model_size']:
            report_content += f"""
### 模型大小统计
- 均值: {comparison_result['statistics']['model_size']['mean']:.2f}MB
- 标准差: {comparison_result['statistics']['model_size']['std']:.2f}MB
- 最小值: {comparison_result['statistics']['model_size']['min']:.2f}MB
- 最大值: {comparison_result['statistics']['model_size']['max']:.2f}MB
"""
        
        report_content += f"""

## 建议

基于以上对比结果，我们提供以下建议：

1. **生产环境部署**: 推荐使用 '{comparison_result['best_models']['overall_best']}' 模型，综合表现最佳
2. **高精度要求**: 如果对准确率要求极高，建议使用 '{comparison_result['best_models']['highest_accuracy']}' 模型
3. **实时推理**: 如果对推理速度有严格要求，建议使用 '{comparison_result['best_models']['fastest_inference']}' 模型
4. **资源受限环境**: 如果存储空间有限，建议使用 '{comparison_result['best_models']['smallest_model']}' 模型

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Comparison report saved to {output_file}")
        return str(output_file)
    
    def ab_test(self, 
               model_a_results: Dict[str, Any],
               model_b_results: Dict[str, Any],
               significance_level: float = 0.05) -> Dict[str, Any]:
        """进行A/B测试
        
        Args:
            model_a_results: 模型A的评估结果
            model_b_results: 模型B的评估结果
            significance_level: 显著性水平
            
        Returns:
            A/B测试结果
        """
        from scipy import stats
        
        # 提取关键指标
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        test_results = {
            'model_a': model_a_results['model_name'],
            'model_b': model_b_results['model_name'],
            'significance_level': significance_level,
            'test_time': datetime.now().isoformat(),
            'comparisons': {}
        }
        
        for metric in metrics:
            value_a = model_a_results['metrics'][metric]
            value_b = model_b_results['metrics'][metric]
            
            if value_a is not None and value_b is not None:
                # 计算差异
                difference = value_b - value_a
                relative_improvement = (difference / value_a) * 100 if value_a != 0 else 0
                
                # 简单的统计显著性检验（需要更多数据点进行准确检验）
                is_significant = abs(relative_improvement) > 5  # 简化判断
                
                test_results['comparisons'][metric] = {
                    'model_a_value': value_a,
                    'model_b_value': value_b,
                    'difference': difference,
                    'relative_improvement_percent': relative_improvement,
                    'is_significant': is_significant,
                    'better_model': model_b_results['model_name'] if value_b > value_a else model_a_results['model_name']
                }
        
        # 综合判断
        significant_improvements = sum(1 for comp in test_results['comparisons'].values() 
                                     if comp['is_significant'] and comp['difference'] > 0)
        total_comparisons = len(test_results['comparisons'])
        
        test_results['summary'] = {
            'significant_improvements': significant_improvements,
            'total_comparisons': total_comparisons,
            'improvement_ratio': significant_improvements / total_comparisons if total_comparisons > 0 else 0,
            'recommendation': 'Model B' if significant_improvements > total_comparisons / 2 else 'Model A'
        }
        
        return test_results
