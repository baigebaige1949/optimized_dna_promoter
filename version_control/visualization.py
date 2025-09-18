"""
版本控制可视化功能
实现训练曲线对比、指标可视化等功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class VersionControlVisualizer:
    """版本控制可视化器"""
    
    def __init__(self, output_dir: str = "./visualizations"):
        """初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置绘图风格
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_training_curves(self, 
                           experiments_data: List[Dict[str, Any]], 
                           metrics: List[str] = ['loss', 'accuracy'],
                           save_path: str = None) -> str:
        """绘制训练曲线对比图
        
        Args:
            experiments_data: 实验数据列表
            metrics: 要绘制的指标
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = self.output_dir / "training_curves.png"
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for exp_data in experiments_data:
                exp_name = exp_data.get('name', 'Unknown')
                history = exp_data.get('metrics_history', {})
                
                if metric in history and history[metric]:
                    epochs = list(range(1, len(history[metric]) + 1))
                    values = history[metric]
                    ax.plot(epochs, values, label=f"{exp_name} - {metric}", marker='o')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} 训练曲线对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_model_comparison_radar(self, 
                                  comparison_data: Dict[str, Any],
                                  save_path: str = None) -> str:
        """绘制模型对比雷达图
        
        Args:
            comparison_data: 对比数据
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = self.output_dir / "model_comparison_radar.png"
        
        # 提取数据
        models_data = comparison_data.get('summary_table', [])
        if not models_data:
            return str(save_path)
        
        # 选择要展示的指标
        metrics = ['accuracy', 'f1_score', 'samples_per_second']
        available_metrics = [m for m in metrics if any(pd.notna(d.get(m)) for d in models_data)]
        
        if not available_metrics:
            return str(save_path)
        
        # 准备数据
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for model_data in models_data:
            model_name = model_data.get('name', 'Unknown')
            values = []
            
            for metric in available_metrics:
                value = model_data.get(metric, 0)
                if pd.isna(value):
                    value = 0
                # 归一化到0-1范围
                all_values = [d.get(metric, 0) for d in models_data if pd.notna(d.get(metric))]
                if all_values and max(all_values) > min(all_values):
                    normalized_value = (value - min(all_values)) / (max(all_values) - min(all_values))
                else:
                    normalized_value = 0.5
                values.append(normalized_value)
            
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('模型性能对比雷达图', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_performance_metrics_heatmap(self, 
                                       comparison_data: Dict[str, Any],
                                       save_path: str = None) -> str:
        """绘制性能指标热力图
        
        Args:
            comparison_data: 对比数据
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = self.output_dir / "performance_heatmap.png"
        
        models_data = comparison_data.get('summary_table', [])
        if not models_data:
            return str(save_path)
        
        # 创建DataFrame
        df = pd.DataFrame(models_data)
        
        # 选择数值型列
        numeric_columns = ['accuracy', 'precision', 'recall', 'f1_score', 
                          'inference_time', 'samples_per_second', 'parameters', 'model_size_mb']
        available_columns = [col for col in numeric_columns if col in df.columns and df[col].notna().any()]
        
        if not available_columns:
            return str(save_path)
        
        # 准备热力图数据
        heatmap_data = df.set_index('name')[available_columns]
        
        # 归一化数据（0-1范围）
        normalized_data = heatmap_data.copy()
        for col in available_columns:
            col_data = heatmap_data[col].dropna()
            if len(col_data) > 0 and col_data.max() != col_data.min():
                normalized_data[col] = (heatmap_data[col] - col_data.min()) / (col_data.max() - col_data.min())
            else:
                normalized_data[col] = 0.5
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(normalized_data, 
                   annot=True, 
                   cmap='YlOrRd', 
                   fmt='.3f',
                   cbar_kws={'label': 'Normalized Score'})
        
        plt.title('模型性能指标热力图', fontsize=16, pad=20)
        plt.xlabel('Performance Metrics', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_model_size_vs_performance(self, 
                                     comparison_data: Dict[str, Any],
                                     performance_metric: str = 'accuracy',
                                     save_path: str = None) -> str:
        """绘制模型大小 vs 性能散点图
        
        Args:
            comparison_data: 对比数据
            performance_metric: 性能指标
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = self.output_dir / f"model_size_vs_{performance_metric}.png"
        
        models_data = comparison_data.get('summary_table', [])
        if not models_data:
            return str(save_path)
        
        # 提取数据
        model_names = []
        model_sizes = []
        performance_values = []
        
        for model_data in models_data:
            if (pd.notna(model_data.get('model_size_mb')) and 
                pd.notna(model_data.get(performance_metric))):
                model_names.append(model_data.get('name', 'Unknown'))
                model_sizes.append(model_data.get('model_size_mb'))
                performance_values.append(model_data.get(performance_metric))
        
        if not model_names:
            return str(save_path)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(model_sizes, performance_values, 
                            s=100, alpha=0.7, c=range(len(model_names)), 
                            cmap='viridis')
        
        # 添加模型名称标注
        for i, name in enumerate(model_names):
            plt.annotate(name, (model_sizes[i], performance_values[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        plt.xlabel('Model Size (MB)', fontsize=12)
        plt.ylabel(f'{performance_metric.replace("_", " ").title()}', fontsize=12)
        plt.title(f'模型大小 vs {performance_metric.replace("_", " ").title()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_comprehensive_report(self, 
                                  comparison_data: Dict[str, Any],
                                  experiments_data: List[Dict[str, Any]] = None,
                                  report_name: str = None) -> Dict[str, str]:
        """创建综合可视化报告
        
        Args:
            comparison_data: 对比数据
            experiments_data: 实验数据
            report_name: 报告名称
            
        Returns:
            生成的图表文件路径字典
        """
        if report_name is None:
            report_name = "comprehensive_report"
        
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)
        
        generated_plots = {}
        
        # 1. 训练曲线（如果有实验数据）
        if experiments_data:
            training_curves_path = report_dir / "training_curves.png"
            generated_plots['training_curves'] = self.plot_training_curves(
                experiments_data, save_path=training_curves_path
            )
        
        # 2. 雷达图
        radar_path = report_dir / "model_comparison_radar.png"
        generated_plots['radar_chart'] = self.plot_model_comparison_radar(
            comparison_data, save_path=radar_path
        )
        
        # 3. 热力图
        heatmap_path = report_dir / "performance_heatmap.png"
        generated_plots['heatmap'] = self.plot_performance_metrics_heatmap(
            comparison_data, save_path=heatmap_path
        )
        
        # 4. 模型大小vs性能散点图
        scatter_path = report_dir / "model_size_vs_performance.png"
        generated_plots['scatter_plot'] = self.plot_model_size_vs_performance(
            comparison_data, save_path=scatter_path
        )
        
        # 5. 生成HTML报告索引
        html_content = self._generate_html_report(generated_plots, comparison_data)
        html_path = report_dir / "index.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        generated_plots['html_report'] = str(html_path)
        
        return generated_plots
    
    def _generate_html_report(self, plots: Dict[str, str], comparison_data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>模型版本对比报告</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .section {{ margin-bottom: 40px; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .chart img {{ max-width: 100%; height: auto; }}
        .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>模型版本对比报告</h1>
        <p>生成时间: {comparison_data.get('comparison_time', 'Unknown')}</p>
    </div>
    
    <div class="section">
        <h2>最佳模型总结</h2>
        <div class="summary">
            <p><strong>整体最佳:</strong> {comparison_data.get('best_models', {}).get('overall_best', 'N/A')}</p>
            <p><strong>准确率最高:</strong> {comparison_data.get('best_models', {}).get('highest_accuracy', 'N/A')}</p>
            <p><strong>推理最快:</strong> {comparison_data.get('best_models', {}).get('fastest_inference', 'N/A')}</p>
            <p><strong>模型最小:</strong> {comparison_data.get('best_models', {}).get('smallest_model', 'N/A')}</p>
        </div>
    </div>
    """
        
        # 添加图表
        for plot_name, plot_path in plots.items():
            if plot_name != 'html_report' and plot_path:
                plot_filename = Path(plot_path).name
                html_template += f"""
    <div class="section">
        <h2>{plot_name.replace('_', ' ').title()}</h2>
        <div class="chart">
            <img src="{plot_filename}" alt="{plot_name}">
        </div>
    </div>
                """
        
        html_template += """
</body>
</html>
        """
        
        return html_template


def create_visualizer(output_dir: str = "./visualizations") -> VersionControlVisualizer:
    """创建可视化器的工厂函数"""
    return VersionControlVisualizer(output_dir)
