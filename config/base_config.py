"""基础配置类"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class BaseConfig:
    """基础配置类，包含项目的基本设置"""
    
    # 项目基本信息
    project_name: str = "dna_promoter_generation"
    version: str = "2.0.0"
    
    # 设备和资源配置
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    batch_size: int = 32
    
    # 文件路径配置
    data_dir: str = "data"
    output_dir: str = "outputs"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # DNA序列配置
    sequence_length: int = 200
    nucleotides: List[str] = field(default_factory=lambda: ['A', 'T', 'G', 'C'])
    
    # 随机种子
    random_seed: int = 42
    
    # 日志级别
    log_level: str = "INFO"
    
    # 是否启用调试模式
    debug: bool = False
    
    def __post_init__(self):
        """初始化后的处理"""
        # 自动检测设备
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 创建目录
        for dir_path in [self.data_dir, self.output_dir, self.model_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """保存配置到YAML文件"""
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def get_device(self) -> torch.device:
        """获取PyTorch设备对象"""
        return torch.device(self.device)
