"""模型版本管理器

实现模型版本的保存、加载、管理等功能
"""

import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch
from pathlib import Path


class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, base_dir: str = "./model_versions"):
        """初始化版本管理器
        
        Args:
            base_dir: 版本存储基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.versions_file = self.base_dir / "versions.json"
        self._load_versions_index()
    
    def _load_versions_index(self):
        """加载版本索引"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r', encoding='utf-8') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def _save_versions_index(self):
        """保存版本索引"""
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)
    
    def save_version(self, 
                    model: torch.nn.Module,
                    version_name: str,
                    metadata: Optional[Dict[str, Any]] = None,
                    description: str = "",
                    auto_increment: bool = False) -> str:
        """保存模型版本
        
        Args:
            model: 要保存的模型
            version_name: 版本名称
            metadata: 版本元数据
            description: 版本描述
            auto_increment: 是否自动递增版本号
            
        Returns:
            实际保存的版本名称
        """
        if auto_increment:
            base_name = version_name
            counter = 1
            while version_name in self.versions:
                version_name = f"{base_name}_v{counter}"
                counter += 1
        
        version_dir = self.base_dir / version_name
        version_dir.mkdir(exist_ok=True)
        
        # 保存模型权重
        model_path = version_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # 保存模型结构信息
        model_info = {
            'class_name': model.__class__.__name__,
            'model_type': type(model).__name__,
            'parameters_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # 保存完整模型（可选）
        try:
            torch.save(model, version_dir / "complete_model.pth")
        except Exception as e:
            print(f"Warning: Could not save complete model: {e}")
        
        # 保存版本信息
        version_info = {
            'version_name': version_name,
            'created_at': datetime.now().isoformat(),
            'description': description,
            'model_info': model_info,
            'metadata': metadata or {},
            'files': {
                'model_weights': str(model_path.relative_to(self.base_dir)),
                'model_info': f"{version_name}/info.json"
            }
        }
        
        # 保存详细信息文件
        info_path = version_dir / "info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        
        # 更新版本索引
        self.versions[version_name] = version_info
        self._save_versions_index()
        
        print(f"Model version '{version_name}' saved successfully")
        return version_name
    
    def load_version(self, version_name: str) -> Dict[str, Any]:
        """加载模型版本
        
        Args:
            version_name: 版本名称
            
        Returns:
            包含模型权重和信息的字典
        """
        if version_name not in self.versions:
            raise ValueError(f"Version '{version_name}' not found")
        
        version_dir = self.base_dir / version_name
        version_info = self.versions[version_name]
        
        result = {
            'version_info': version_info,
            'weights_path': version_dir / "model.pth"
        }
        
        # 尝试加载完整模型
        complete_model_path = version_dir / "complete_model.pth"
        if complete_model_path.exists():
            try:
                result['complete_model'] = torch.load(complete_model_path)
            except Exception as e:
                print(f"Warning: Could not load complete model: {e}")
        
        return result
    
    def load_model_weights(self, version_name: str, model: torch.nn.Module) -> torch.nn.Module:
        """加载模型权重到指定模型
        
        Args:
            version_name: 版本名称
            model: 目标模型
            
        Returns:
            加载权重后的模型
        """
        version_data = self.load_version(version_name)
        weights_path = version_data['weights_path']
        
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"Loaded weights from version '{version_name}'")
        return model
    
    def list_versions(self, sort_by: str = 'created_at') -> List[Dict[str, Any]]:
        """列出所有版本
        
        Args:
            sort_by: 排序字段
            
        Returns:
            版本列表
        """
        versions_list = list(self.versions.values())
        
        if sort_by == 'created_at':
            versions_list.sort(key=lambda x: x['created_at'], reverse=True)
        elif sort_by == 'name':
            versions_list.sort(key=lambda x: x['version_name'])
        
        return versions_list
    
    def delete_version(self, version_name: str, confirm: bool = False):
        """删除版本
        
        Args:
            version_name: 版本名称
            confirm: 是否确认删除
        """
        if not confirm:
            print(f"Warning: This will permanently delete version '{version_name}'")
            print("Use confirm=True to proceed")
            return
        
        if version_name not in self.versions:
            raise ValueError(f"Version '{version_name}' not found")
        
        version_dir = self.base_dir / version_name
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        del self.versions[version_name]
        self._save_versions_index()
        
        print(f"Version '{version_name}' deleted successfully")
    
    def get_version_info(self, version_name: str) -> Dict[str, Any]:
        """获取版本详细信息
        
        Args:
            version_name: 版本名称
            
        Returns:
            版本信息
        """
        if version_name not in self.versions:
            raise ValueError(f"Version '{version_name}' not found")
        
        return self.versions[version_name]
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """比较两个版本
        
        Args:
            version1: 版本1名称
            version2: 版本2名称
            
        Returns:
            比较结果
        """
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        comparison = {
            'version1': info1,
            'version2': info2,
            'differences': {
                'parameters_count': info1['model_info']['parameters_count'] - info2['model_info']['parameters_count'],
                'creation_time_diff': (datetime.fromisoformat(info1['created_at']) - 
                                     datetime.fromisoformat(info2['created_at'])).total_seconds()
            }
        }
        
        return comparison
    
    def create_checkpoint(self, 
                         model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         loss: float,
                         metrics: Dict[str, float],
                         checkpoint_name: str = None) -> str:
        """创建训练检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 训练轮数
            loss: 损失值
            metrics: 评估指标
            checkpoint_name: 检查点名称
            
        Returns:
            检查点名称
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_dir = self.base_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存检查点数据
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        torch.save(checkpoint_data, checkpoint_path)
        
        # 保存检查点信息
        info = {
            'checkpoint_name': checkpoint_name,
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'created_at': datetime.now().isoformat(),
            'path': str(checkpoint_path)
        }
        
        info_path = checkpoint_dir / "info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"Checkpoint '{checkpoint_name}' saved successfully")
        return checkpoint_name
