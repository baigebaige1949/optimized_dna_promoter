"""设备管理器"""

import torch
from typing import Optional, List, Dict, Any
from .logger import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """设备管理器，负责GPU/CPU的选择和管理"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = self._select_device(device)
        self.device_info = self._get_device_info()
        
        logger.info(f"Device manager initialized - Using: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU Info: {self.device_info}")
    
    def _select_device(self, device: Optional[str] = None) -> torch.device:
        """选择设备"""
        if device is None or device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def _get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'device_type': self.device.type,
            'device_name': str(self.device)
        }
        
        if self.device.type == 'cuda':
            gpu_id = self.device.index or 0
            info.update({
                'gpu_id': gpu_id,
                'gpu_name': torch.cuda.get_device_name(gpu_id),
                'memory_total': torch.cuda.get_device_properties(gpu_id).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(gpu_id),
                'memory_cached': torch.cuda.memory_reserved(gpu_id)
            })
        
        return info
    
    def get_available_devices(self) -> List[str]:
        """获取可用设备列表"""
        devices = ['cpu']
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f'cuda:{i}')
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps')
        
        return devices
    
    def move_to_device(self, obj) -> Any:
        """将对象移动到指定设备"""
        if hasattr(obj, 'to'):
            return obj.to(self.device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)([self.move_to_device(item) for item in obj])
        elif isinstance(obj, dict):
            return {key: self.move_to_device(value) for key, value in obj.items()}
        else:
            return obj
    
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        info = {}
        
        if self.device.type == 'cuda':
            gpu_id = self.device.index or 0
            info['memory_allocated'] = torch.cuda.memory_allocated(gpu_id) / 1024**2  # MB
            info['memory_cached'] = torch.cuda.memory_reserved(gpu_id) / 1024**2      # MB
            info['memory_total'] = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2  # MB
            info['memory_free'] = info['memory_total'] - info['memory_allocated']
        else:
            # CPU内存信息（需要psutil）
            try:
                import psutil
                memory = psutil.virtual_memory()
                info['memory_total'] = memory.total / 1024**2
                info['memory_available'] = memory.available / 1024**2
                info['memory_used'] = memory.used / 1024**2
            except ImportError:
                logger.debug("psutil not available, cannot get CPU memory info")
        
        return info
    
    def clear_cache(self) -> None:
        """清理缓存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
    
    def set_device(self, device: str) -> None:
        """设置新设备"""
        old_device = self.device
        self.device = torch.device(device)
        self.device_info = self._get_device_info()
        logger.info(f"Device changed from {old_device} to {self.device}")
    
    def is_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        return self.device.type in ['cuda', 'mps']
    
    def get_optimal_batch_size(self, model_memory_mb: float, sequence_length: int = 200) -> int:
        """估算最优批次大小"""
        if self.device.type != 'cuda':
            return 32  # CPU默认批次大小
        
        memory_info = self.get_memory_info()
        available_memory = memory_info.get('memory_free', 1000)  # MB
        
        # 保留一些内存用于其他操作
        usable_memory = available_memory * 0.7
        
        # 估算每个样本的内存消耗
        memory_per_sample = model_memory_mb + sequence_length * 0.001  # 粗略估算
        
        optimal_batch_size = max(1, int(usable_memory / memory_per_sample))
        optimal_batch_size = min(optimal_batch_size, 128)  # 最大批次大小限制
        
        logger.info(f"Estimated optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def monitor_memory_usage(self, operation: str) -> None:
        """监控内存使用情况"""
        memory_info = self.get_memory_info()
        
        if self.device.type == 'cuda':
            logger.info(
                f"{operation} - GPU Memory: "
                f"Allocated: {memory_info.get('memory_allocated', 0):.1f}MB, "
                f"Cached: {memory_info.get('memory_cached', 0):.1f}MB, "
                f"Free: {memory_info.get('memory_free', 0):.1f}MB"
            )
        else:
            logger.info(
                f"{operation} - CPU Memory: "
                f"Available: {memory_info.get('memory_available', 0):.1f}MB"
            )
    
    def __str__(self) -> str:
        return f"DeviceManager(device={self.device}, info={self.device_info})"
