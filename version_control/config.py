"""版本控制系统配置文件

用于配置系统的各种参数和设置
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class VersionControlConfig:
    """版本控制配置类"""
    
    # 基础目录配置
    base_dir: str = "./version_control_workspace"
    versions_dir: str = "versions"
    experiments_dir: str = "experiments"
    results_dir: str = "results"
    visualizations_dir: str = "visualizations"
    
    # 版本管理配置
    auto_versioning: bool = True
    auto_increment_versions: bool = True
    max_versions_to_keep: int = 50
    version_name_format: str = "{model_name}_v{version}"
    
    # 检查点配置
    auto_checkpoint: bool = True
    checkpoint_interval: int = 10  # 每多少轮保存一次
    max_checkpoints_per_experiment: int = 20
    checkpoint_on_best_metric: bool = True
    best_metric_name: str = "val_accuracy"
    
    # 实验跟踪配置
    log_hyperparameters: bool = True
    log_metrics_every_epoch: bool = True
    log_system_info: bool = True
    auto_export_on_completion: bool = True
    
    # 性能对比配置
    default_metrics: List[str] = None
    performance_threshold: float = 0.05  # 性能差异阈值
    auto_generate_reports: bool = True
    include_model_complexity: bool = True
    
    # 可视化配置
    figure_format: str = "png"
    figure_dpi: int = 300
    figure_size: tuple = (12, 8)
    color_scheme: str = "viridis"
    chinese_font: str = "SimHei"
    
    # 数据保存配置
    save_format: str = "json"  # json, pickle, hdf5
    compression: bool = True
    backup_important_data: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        if self.default_metrics is None:
            self.default_metrics = ["accuracy", "precision", "recall", "f1_score"]
        
        # 创建目录结构
        self.create_directories()
    
    def create_directories(self):
        """创建必要的目录结构"""
        base_path = Path(self.base_dir)
        
        directories = [
            base_path / self.versions_dir,
            base_path / self.versions_dir / "checkpoints",
            base_path / self.experiments_dir,
            base_path / self.results_dir,
            base_path / self.results_dir / "reports",
            base_path / self.visualizations_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_full_path(self, subdir: str) -> str:
        """获取完整路径"""
        return str(Path(self.base_dir) / subdir)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "base_dir": self.base_dir,
            "versions_dir": self.versions_dir,
            "experiments_dir": self.experiments_dir,
            "results_dir": self.results_dir,
            "visualizations_dir": self.visualizations_dir,
            "auto_versioning": self.auto_versioning,
            "auto_increment_versions": self.auto_increment_versions,
            "max_versions_to_keep": self.max_versions_to_keep,
            "version_name_format": self.version_name_format,
            "auto_checkpoint": self.auto_checkpoint,
            "checkpoint_interval": self.checkpoint_interval,
            "max_checkpoints_per_experiment": self.max_checkpoints_per_experiment,
            "checkpoint_on_best_metric": self.checkpoint_on_best_metric,
            "best_metric_name": self.best_metric_name,
            "log_hyperparameters": self.log_hyperparameters,
            "log_metrics_every_epoch": self.log_metrics_every_epoch,
            "log_system_info": self.log_system_info,
            "auto_export_on_completion": self.auto_export_on_completion,
            "default_metrics": self.default_metrics,
            "performance_threshold": self.performance_threshold,
            "auto_generate_reports": self.auto_generate_reports,
            "include_model_complexity": self.include_model_complexity,
            "figure_format": self.figure_format,
            "figure_dpi": self.figure_dpi,
            "figure_size": self.figure_size,
            "color_scheme": self.color_scheme,
            "chinese_font": self.chinese_font,
            "save_format": self.save_format,
            "compression": self.compression,
            "backup_important_data": self.backup_important_data
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'VersionControlConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'VersionControlConfig':
        """从文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# 默认配置实例
DEFAULT_CONFIG = VersionControlConfig()

# 高性能配置（适用于大规模实验）
HIGH_PERFORMANCE_CONFIG = VersionControlConfig(
    base_dir="./high_performance_workspace",
    checkpoint_interval=5,  # 更频繁的检查点
    max_versions_to_keep=100,
    max_checkpoints_per_experiment=50,
    figure_dpi=150,  # 降低图片质量以节省空间
    compression=True,
    backup_important_data=True
)

# 快速原型配置（适用于快速实验）
RAPID_PROTOTYPING_CONFIG = VersionControlConfig(
    base_dir="./rapid_prototyping",
    auto_versioning=False,  # 关闭自动版本管理
    auto_checkpoint=False,  # 关闭自动检查点
    max_versions_to_keep=10,
    log_system_info=False,
    auto_generate_reports=False,
    figure_dpi=100,  # 低质量图片
    backup_important_data=False
)

# 生产环境配置
PRODUCTION_CONFIG = VersionControlConfig(
    base_dir="./production_workspace",
    auto_versioning=True,
    checkpoint_interval=20,  # 生产环境中检查点间隔更大
    max_versions_to_keep=20,  # 保留较少的版本
    auto_export_on_completion=True,
    backup_important_data=True,
    log_system_info=True,
    include_model_complexity=True
)


def get_config(config_type: str = "default") -> VersionControlConfig:
    """获取指定类型的配置
    
    Args:
        config_type: 配置类型 (default, high_performance, rapid_prototyping, production)
        
    Returns:
        配置对象
    """
    configs = {
        "default": DEFAULT_CONFIG,
        "high_performance": HIGH_PERFORMANCE_CONFIG,
        "rapid_prototyping": RAPID_PROTOTYPING_CONFIG,
        "production": PRODUCTION_CONFIG
    }
    
    if config_type not in configs:
        print(f"Warning: Unknown config type '{config_type}', using default config")
        return DEFAULT_CONFIG
    
    return configs[config_type]


def create_custom_config(**kwargs) -> VersionControlConfig:
    """创建自定义配置
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        自定义配置对象
    """
    return VersionControlConfig(**kwargs)


if __name__ == "__main__":
    # 演示配置使用
    print("版本控制系统配置演示")
    print("=" * 40)
    
    # 使用默认配置
    default_config = get_config("default")
    print(f"默认配置基础目录: {default_config.base_dir}")
    print(f"自动版本管理: {default_config.auto_versioning}")
    
    # 使用高性能配置
    hp_config = get_config("high_performance")
    print(f"\n高性能配置检查点间隔: {hp_config.checkpoint_interval}")
    print(f"最大版本数: {hp_config.max_versions_to_keep}")
    
    # 创建自定义配置
    custom_config = create_custom_config(
        base_dir="./my_experiments",
        auto_versioning=True,
        checkpoint_interval=15
    )
    print(f"\n自定义配置基础目录: {custom_config.base_dir}")
    
    # 保存配置到文件
    default_config.save_to_file("./default_config.json")
    
    # 从文件加载配置
    loaded_config = VersionControlConfig.load_from_file("./default_config.json")
    print(f"\n从文件加载的配置: {loaded_config.base_dir}")
    
    print("\n配置演示完成！")
