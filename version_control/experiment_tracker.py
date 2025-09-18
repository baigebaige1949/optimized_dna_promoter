"""
实验跟踪和结果记录系统
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import numpy as np


class ExperimentTracker:
    """实验跟踪器"""
    
    def __init__(self, workspace_dir: str = "./experiments"):
        """初始化实验跟踪器
        
        Args:
            workspace_dir: 实验工作空间目录
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        self.experiments_file = self.workspace_dir / "experiments.json"
        self.current_experiment = None
        self._load_experiments()
    
    def _load_experiments(self):
        """加载实验记录"""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r', encoding='utf-8') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}
    
    def _save_experiments(self):
        """保存实验记录"""
        with open(self.experiments_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False, default=str)
    
    def start_experiment(self, 
                        name: str = None,
                        description: str = "",
                        config: Dict[str, Any] = None) -> str:
        """开始新实验
        
        Args:
            name: 实验名称
            description: 实验描述
            config: 实验配置
            
        Returns:
            实验ID
        """
        experiment_id = str(uuid.uuid4())
        
        if name is None:
            name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_info = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'config': config or {},
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'status': 'running',
            'metrics': {},
            'metrics_history': {},
            'artifacts': [],
            'logs': []
        }
        
        # 创建实验目录
        experiment_dir = self.workspace_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # 保存实验配置
        config_file = experiment_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_info['config'], f, indent=2, ensure_ascii=False)
        
        self.experiments[experiment_id] = experiment_info
        self.current_experiment = experiment_id
        self._save_experiments()
        
        print(f"Started experiment '{name}' (ID: {experiment_id})")
        return experiment_id
    
    def end_experiment(self, experiment_id: str = None, status: str = 'completed'):
        """结束实验
        
        Args:
            experiment_id: 实验ID，如果None则使用当前实验
            status: 实验状态
        """
        if experiment_id is None:
            experiment_id = self.current_experiment
        
        if experiment_id and experiment_id in self.experiments:
            self.experiments[experiment_id]['end_time'] = datetime.now().isoformat()
            self.experiments[experiment_id]['status'] = status
            self._save_experiments()
            
            if experiment_id == self.current_experiment:
                self.current_experiment = None
            
            print(f"Ended experiment {experiment_id} with status: {status}")
    
    def log_metric(self, 
                  metric_name: str, 
                  value: float, 
                  step: int = None,
                  experiment_id: str = None):
        """记录指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            step: 步数
            experiment_id: 实验ID
        """
        if experiment_id is None:
            experiment_id = self.current_experiment
        
        if not experiment_id or experiment_id not in self.experiments:
            return
        
        # 更新当前指标
        self.experiments[experiment_id]['metrics'][metric_name] = value
        
        # 更新指标历史
        if metric_name not in self.experiments[experiment_id]['metrics_history']:
            self.experiments[experiment_id]['metrics_history'][metric_name] = []
        
        metric_entry = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        if step is not None:
            metric_entry['step'] = step
        
        self.experiments[experiment_id]['metrics_history'][metric_name].append(metric_entry)
        self._save_experiments()
    
    def log_metrics(self, 
                   metrics: Dict[str, float], 
                   step: int = None,
                   experiment_id: str = None):
        """批量记录指标
        
        Args:
            metrics: 指标字典
            step: 步数
            experiment_id: 实验ID
        """
        for metric_name, value in metrics.items():
            self.log_metric(metric_name, value, step, experiment_id)
    
    def log_hyperparameters(self, 
                          hyperparams: Dict[str, Any], 
                          experiment_id: str = None):
        """记录超参数
        
        Args:
            hyperparams: 超参数字典
            experiment_id: 实验ID
        """
        if experiment_id is None:
            experiment_id = self.current_experiment
        
        if not experiment_id or experiment_id not in self.experiments:
            return
        
        self.experiments[experiment_id]['config'].update(hyperparams)
        
        # 保存到文件
        experiment_dir = self.workspace_dir / experiment_id
        hyperparams_file = experiment_dir / "hyperparameters.json"
        with open(hyperparams_file, 'w', encoding='utf-8') as f:
            json.dump(hyperparams, f, indent=2, ensure_ascii=False, default=str)
        
        self._save_experiments()
    
    def save_artifact(self, 
                     artifact_path: str, 
                     artifact_name: str = None,
                     experiment_id: str = None) -> str:
        """保存实验产物
        
        Args:
            artifact_path: 产物文件路径
            artifact_name: 产物名称
            experiment_id: 实验ID
            
        Returns:
            保存的产物路径
        """
        if experiment_id is None:
            experiment_id = self.current_experiment
        
        if not experiment_id or experiment_id not in self.experiments:
            return artifact_path
        
        if artifact_name is None:
            artifact_name = Path(artifact_path).name
        
        # 创建artifacts目录
        experiment_dir = self.workspace_dir / experiment_id
        artifacts_dir = experiment_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # 复制文件到artifacts目录
        import shutil
        source_path = Path(artifact_path)
        dest_path = artifacts_dir / artifact_name
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            
            # 记录artifact信息
            artifact_info = {
                'name': artifact_name,
                'path': str(dest_path.relative_to(self.workspace_dir)),
                'original_path': str(artifact_path),
                'timestamp': datetime.now().isoformat(),
                'size': dest_path.stat().st_size if dest_path.exists() else 0
            }
            
            self.experiments[experiment_id]['artifacts'].append(artifact_info)
            self._save_experiments()
            
            return str(dest_path)
        
        return artifact_path
    
    def add_log(self, message: str, level: str = 'info', experiment_id: str = None):
        """添加日志
        
        Args:
            message: 日志消息
            level: 日志级别
            experiment_id: 实验ID
        """
        if experiment_id is None:
            experiment_id = self.current_experiment
        
        if not experiment_id or experiment_id not in self.experiments:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        
        self.experiments[experiment_id]['logs'].append(log_entry)
        
        # 限制日志数量
        if len(self.experiments[experiment_id]['logs']) > 1000:
            self.experiments[experiment_id]['logs'] = self.experiments[experiment_id]['logs'][-1000:]
        
        self._save_experiments()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """获取实验信息
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验信息
        """
        return self.experiments.get(experiment_id)
    
    def list_experiments(self, 
                        status: str = None, 
                        sort_by: str = 'start_time') -> List[Dict[str, Any]]:
        """列出实验
        
        Args:
            status: 过滤状态
            sort_by: 排序字段
            
        Returns:
            实验列表
        """
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [exp for exp in experiments if exp.get('status') == status]
        
        if sort_by == 'start_time':
            experiments.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        elif sort_by == 'name':
            experiments.sort(key=lambda x: x.get('name', ''))
        
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """比较实验
        
        Args:
            experiment_ids: 实验ID列表
            
        Returns:
            比较结果
        """
        experiments = [self.experiments[exp_id] for exp_id in experiment_ids if exp_id in self.experiments]
        
        if not experiments:
            return {}
        
        # 收集所有指标
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.get('metrics', {}).keys())
        
        comparison = {
            'comparison_time': datetime.now().isoformat(),
            'experiments': experiments,
            'metrics_comparison': {},
            'best_performers': {}
        }
        
        # 比较指标
        for metric in all_metrics:
            metric_values = {}
            for exp in experiments:
                value = exp.get('metrics', {}).get(metric)
                if value is not None:
                    metric_values[exp['name']] = value
            
            if metric_values:
                comparison['metrics_comparison'][metric] = metric_values
                
                # 找出最佳表现者
                if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    best_exp = max(metric_values.items(), key=lambda x: x[1])
                else:  # loss等，越小越好
                    best_exp = min(metric_values.items(), key=lambda x: x[1])
                
                comparison['best_performers'][metric] = {
                    'experiment': best_exp[0],
                    'value': best_exp[1]
                }
        
        return comparison
    
    def delete_experiment(self, experiment_id: str, confirm: bool = False):
        """删除实验
        
        Args:
            experiment_id: 实验ID
            confirm: 确认删除
        """
        if not confirm:
            print(f"Warning: This will permanently delete experiment '{experiment_id}'")
            print("Use confirm=True to proceed")
            return
        
        if experiment_id not in self.experiments:
            print(f"Experiment '{experiment_id}' not found")
            return
        
        # 删除实验目录
        experiment_dir = self.workspace_dir / experiment_id
        if experiment_dir.exists():
            import shutil
            shutil.rmtree(experiment_dir)
        
        # 从记录中删除
        del self.experiments[experiment_id]
        self._save_experiments()
        
        if experiment_id == self.current_experiment:
            self.current_experiment = None
        
        print(f"Experiment '{experiment_id}' deleted successfully")


def create_experiment_tracker(workspace_dir: str = "./experiments") -> ExperimentTracker:
    """创建实验跟踪器的工厂函数"""
    return ExperimentTracker(workspace_dir)
