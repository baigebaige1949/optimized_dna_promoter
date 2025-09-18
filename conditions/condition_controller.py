#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
条件控制模块
支持温度、pH、氧气含量、细胞生长周期等多种条件的智能管理
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ConditionType(Enum):
    """条件类型枚举"""
    TEMPERATURE = "temperature"  # 温度
    PH = "ph"  # pH值
    OXYGEN = "oxygen"  # 氧气含量
    CELL_CYCLE = "cell_cycle"  # 细胞生长周期
    NUTRIENTS = "nutrients"  # 营养条件
    STRESS = "stress"  # 胁迫条件
    TIME = "time"  # 时间条件
    CONCENTRATION = "concentration"  # 浓度
    LIGHT = "light"  # 光照条件
    OSMOLARITY = "osmolarity"  # 渗透压
    CUSTOM = "custom"  # 自定义条件

@dataclass
class ConditionRange:
    """条件范围定义"""
    min_value: float
    max_value: float
    default_value: float
    unit: str = ""
    description: str = ""
    
    def __post_init__(self):
        if not (self.min_value <= self.default_value <= self.max_value):
            raise ValueError(f"默认值{self.default_value}不在范围[{self.min_value}, {self.max_value}]内")
    
    def normalize(self, value: float) -> float:
        """将值标准化到[0, 1]范围"""
        if value < self.min_value:
            return 0.0
        elif value > self.max_value:
            return 1.0
        else:
            return (value - self.min_value) / (self.max_value - self.min_value)
    
    def denormalize(self, normalized_value: float) -> float:
        """从[0, 1]范围反标准化"""
        return self.min_value + normalized_value * (self.max_value - self.min_value)
    
    def is_valid(self, value: float) -> bool:
        """检查值是否在有效范围内"""
        return self.min_value <= value <= self.max_value

class ConditionDefinition:
    """条件定义类"""
    
    # 预定义的生物学条件范围
    PREDEFINED_CONDITIONS = {
        ConditionType.TEMPERATURE: ConditionRange(
            min_value=4.0, max_value=85.0, default_value=37.0,
            unit="°C", description="培养温度"
        ),
        ConditionType.PH: ConditionRange(
            min_value=1.0, max_value=14.0, default_value=7.4,
            unit="pH", description="酸碱度"
        ),
        ConditionType.OXYGEN: ConditionRange(
            min_value=0.0, max_value=100.0, default_value=21.0,
            unit="%", description="氧气浓度"
        ),
        ConditionType.CELL_CYCLE: ConditionRange(
            min_value=0.0, max_value=24.0, default_value=12.0,
            unit="hours", description="细胞周期时间"
        ),
        ConditionType.NUTRIENTS: ConditionRange(
            min_value=0.0, max_value=100.0, default_value=50.0,
            unit="%", description="营养浓度"
        ),
        ConditionType.STRESS: ConditionRange(
            min_value=0.0, max_value=10.0, default_value=0.0,
            unit="scale", description="胁迫强度"
        ),
        ConditionType.TIME: ConditionRange(
            min_value=0.0, max_value=168.0, default_value=24.0,
            unit="hours", description="时间点"
        ),
        ConditionType.CONCENTRATION: ConditionRange(
            min_value=0.001, max_value=1000.0, default_value=1.0,
            unit="μM", description="分子浓度"
        ),
        ConditionType.LIGHT: ConditionRange(
            min_value=0.0, max_value=2000.0, default_value=200.0,
            unit="μmol/m²/s", description="光照强度"
        ),
        ConditionType.OSMOLARITY: ConditionRange(
            min_value=100.0, max_value=600.0, default_value=300.0,
            unit="mOsm", description="渗透压"
        )
    }
    
    @classmethod
    def get_condition_range(cls, condition_type: ConditionType) -> ConditionRange:
        """获取条件范围"""
        return cls.PREDEFINED_CONDITIONS.get(condition_type)
    
    @classmethod
    def add_custom_condition(cls, name: str, condition_range: ConditionRange) -> None:
        """添加自定义条件"""
        custom_type = ConditionType.CUSTOM
        cls.PREDEFINED_CONDITIONS[f"{custom_type.value}_{name}"] = condition_range
        logger.info(f"添加自定义条件: {name}")

class ConditionVector:
    """条件向量类"""
    
    def __init__(self, conditions: Optional[Dict[str, float]] = None):
        self.conditions = conditions or {}
        self.condition_definitions = ConditionDefinition()
    
    def set_condition(self, condition_type: Union[ConditionType, str], value: float) -> None:
        """设置条件值"""
        key = condition_type.value if isinstance(condition_type, ConditionType) else condition_type
        
        # 验证值的有效性
        if isinstance(condition_type, ConditionType):
            condition_range = self.condition_definitions.get_condition_range(condition_type)
            if condition_range and not condition_range.is_valid(value):
                logger.warning(f"条件值 {value} 超出范围 [{condition_range.min_value}, {condition_range.max_value}]")
        
        self.conditions[key] = value
        logger.debug(f"设置条件 {key} = {value}")
    
    def get_condition(self, condition_type: Union[ConditionType, str], 
                     default: Optional[float] = None) -> Optional[float]:
        """获取条件值"""
        key = condition_type.value if isinstance(condition_type, ConditionType) else condition_type
        
        if key in self.conditions:
            return self.conditions[key]
        elif default is not None:
            return default
        elif isinstance(condition_type, ConditionType):
            condition_range = self.condition_definitions.get_condition_range(condition_type)
            return condition_range.default_value if condition_range else None
        else:
            return None
    
    def normalize_conditions(self) -> Dict[str, float]:
        """标准化所有条件到[0, 1]范围"""
        normalized = {}
        
        for key, value in self.conditions.items():
            # 尝试找到对应的条件类型
            condition_type = None
            for ct in ConditionType:
                if ct.value == key:
                    condition_type = ct
                    break
            
            if condition_type:
                condition_range = self.condition_definitions.get_condition_range(condition_type)
                if condition_range:
                    normalized[key] = condition_range.normalize(value)
                else:
                    normalized[key] = value  # 无法标准化时保持原值
            else:
                normalized[key] = value  # 自定义条件保持原值
        
        return normalized
    
    def to_tensor(self, condition_order: Optional[List[str]] = None) -> torch.Tensor:
        """转换为张量表示"""
        if condition_order is None:
            condition_order = sorted(self.conditions.keys())
        
        values = []
        for key in condition_order:
            value = self.conditions.get(key, 0.0)
            values.append(value)
        
        return torch.tensor(values, dtype=torch.float32)
    
    def from_tensor(self, tensor: torch.Tensor, condition_order: List[str]) -> None:
        """从张量恢复条件"""
        if len(tensor) != len(condition_order):
            raise ValueError(f"张量长度 {len(tensor)} 与条件顺序长度 {len(condition_order)} 不匹配")
        
        self.conditions = {}
        for i, key in enumerate(condition_order):
            self.conditions[key] = tensor[i].item()
    
    def copy(self) -> 'ConditionVector':
        """复制条件向量"""
        return ConditionVector(self.conditions.copy())
    
    def merge(self, other: 'ConditionVector', override: bool = False) -> 'ConditionVector':
        """合并两个条件向量"""
        result = self.copy()
        
        for key, value in other.conditions.items():
            if key not in result.conditions or override:
                result.conditions[key] = value
        
        return result
    
    def __len__(self) -> int:
        return len(self.conditions)
    
    def __str__(self) -> str:
        return f"ConditionVector({self.conditions})"

class ConditionConstraint(ABC):
    """条件约束基类"""
    
    @abstractmethod
    def validate(self, condition_vector: ConditionVector) -> Tuple[bool, str]:
        """验证条件是否满足约束"""
        pass
    
    @abstractmethod
    def adjust(self, condition_vector: ConditionVector) -> ConditionVector:
        """调整条件以满足约束"""
        pass

class RangeConstraint(ConditionConstraint):
    """范围约束"""
    
    def __init__(self, condition_type: Union[ConditionType, str], 
                 min_value: float, max_value: float):
        self.condition_type = condition_type
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, condition_vector: ConditionVector) -> Tuple[bool, str]:
        """验证范围约束"""
        key = self.condition_type.value if isinstance(self.condition_type, ConditionType) else self.condition_type
        value = condition_vector.get_condition(key)
        
        if value is None:
            return True, ""  # 未设置的条件不验证
        
        if self.min_value <= value <= self.max_value:
            return True, ""
        else:
            return False, f"条件 {key} 值 {value} 超出范围 [{self.min_value}, {self.max_value}]"
    
    def adjust(self, condition_vector: ConditionVector) -> ConditionVector:
        """调整到范围内"""
        result = condition_vector.copy()
        key = self.condition_type.value if isinstance(self.condition_type, ConditionType) else self.condition_type
        value = result.get_condition(key)
        
        if value is not None:
            adjusted_value = max(self.min_value, min(self.max_value, value))
            result.set_condition(key, adjusted_value)
        
        return result

class BiologicalCompatibilityConstraint(ConditionConstraint):
    """生物学兼容性约束"""
    
    def __init__(self):
        # 定义一些基本的生物学约束规则
        self.rules = [
            # 极端pH和高温不兼容
            lambda cv: not (cv.get_condition(ConditionType.PH, 7.0) < 3.0 and 
                          cv.get_condition(ConditionType.TEMPERATURE, 37.0) > 60.0),
            # 无氧条件下某些代谢路径不活跃
            lambda cv: not (cv.get_condition(ConditionType.OXYGEN, 21.0) < 1.0 and 
                          cv.get_condition(ConditionType.STRESS, 0.0) > 8.0),
        ]
    
    def validate(self, condition_vector: ConditionVector) -> Tuple[bool, str]:
        """验证生物学兼容性"""
        for i, rule in enumerate(self.rules):
            if not rule(condition_vector):
                return False, f"违反生物学兼容性规则 {i+1}"
        
        return True, ""
    
    def adjust(self, condition_vector: ConditionVector) -> ConditionVector:
        """调整以满足生物学兼容性"""
        result = condition_vector.copy()
        
        # 简单的调整策略：如果违反规则，则调整到安全值
        is_valid, _ = self.validate(result)
        if not is_valid:
            # 调整到相对安全的条件
            result.set_condition(ConditionType.TEMPERATURE, 37.0)
            result.set_condition(ConditionType.PH, 7.4)
            result.set_condition(ConditionType.OXYGEN, 21.0)
            result.set_condition(ConditionType.STRESS, 0.0)
            logger.info("调整条件以满足生物学兼容性")
        
        return result

class ConditionController:
    """条件控制器主类"""
    
    def __init__(self):
        self.constraints: List[ConditionConstraint] = []
        self.default_conditions = ConditionVector()
        self.condition_history: List[ConditionVector] = []
        
        # 添加默认约束
        self.add_constraint(BiologicalCompatibilityConstraint())
        
        # 为预定义条件添加范围约束
        for condition_type, condition_range in ConditionDefinition.PREDEFINED_CONDITIONS.items():
            if isinstance(condition_type, ConditionType):
                self.add_constraint(RangeConstraint(
                    condition_type, condition_range.min_value, condition_range.max_value
                ))
        
        logger.info("条件控制器初始化完成")
    
    def add_constraint(self, constraint: ConditionConstraint) -> None:
        """添加约束"""
        self.constraints.append(constraint)
        logger.debug(f"添加约束: {type(constraint).__name__}")
    
    def set_default_conditions(self, conditions: Dict[Union[ConditionType, str], float]) -> None:
        """设置默认条件"""
        self.default_conditions = ConditionVector()
        for condition_type, value in conditions.items():
            self.default_conditions.set_condition(condition_type, value)
        
        logger.info(f"设置默认条件: {len(conditions)}个")
    
    def create_condition_vector(self, conditions: Optional[Dict[Union[ConditionType, str], float]] = None) -> ConditionVector:
        """创建条件向量"""
        # 从默认条件开始
        result = self.default_conditions.copy()
        
        # 应用用户指定的条件
        if conditions:
            user_conditions = ConditionVector()
            for condition_type, value in conditions.items():
                user_conditions.set_condition(condition_type, value)
            result = result.merge(user_conditions, override=True)
        
        return result
    
    def validate_conditions(self, condition_vector: ConditionVector) -> Tuple[bool, List[str]]:
        """验证条件向量"""
        errors = []
        
        for constraint in self.constraints:
            is_valid, error = constraint.validate(condition_vector)
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors
    
    def adjust_conditions(self, condition_vector: ConditionVector) -> ConditionVector:
        """调整条件以满足所有约束"""
        result = condition_vector.copy()
        
        # 迭代应用所有约束
        max_iterations = 10
        for iteration in range(max_iterations):
            adjusted = False
            
            for constraint in self.constraints:
                is_valid, _ = constraint.validate(result)
                if not is_valid:
                    result = constraint.adjust(result)
                    adjusted = True
            
            if not adjusted:
                break  # 没有更多调整需要
            
            logger.debug(f"约束调整迭代 {iteration + 1}")
        
        return result
    
    def fill_missing_conditions(self, condition_vector: ConditionVector, 
                               biological_context: Optional[str] = None) -> ConditionVector:
        """智能填充缺失的条件"""
        result = condition_vector.copy()
        
        # 基于生物学上下文的智能填充
        context_defaults = self._get_context_defaults(biological_context)
        
        # 填充所有预定义条件的默认值
        for condition_type, condition_range in ConditionDefinition.PREDEFINED_CONDITIONS.items():
            if isinstance(condition_type, ConditionType):
                key = condition_type.value
                if result.get_condition(key) is None:
                    # 优先使用上下文默认值
                    default_value = context_defaults.get(key, condition_range.default_value)
                    result.set_condition(key, default_value)
                    logger.debug(f"填充缺失条件 {key} = {default_value}")
        
        return result
    
    def _get_context_defaults(self, biological_context: Optional[str]) -> Dict[str, float]:
        """根据生物学上下文获取默认值"""
        if not biological_context:
            return {}
        
        context = biological_context.lower()
        
        # 预定义的上下文默认值
        context_map = {
            'e.coli': {
                'temperature': 37.0,
                'ph': 7.0,
                'oxygen': 21.0,
                'nutrients': 80.0
            },
            'yeast': {
                'temperature': 30.0,
                'ph': 6.0,
                'oxygen': 21.0,
                'nutrients': 60.0
            },
            'mammalian': {
                'temperature': 37.0,
                'ph': 7.4,
                'oxygen': 5.0,  # 通常培养在低氧环境
                'nutrients': 70.0
            },
            'plant': {
                'temperature': 25.0,
                'ph': 6.5,
                'light': 200.0,
                'nutrients': 50.0
            },
            'thermophile': {
                'temperature': 70.0,
                'ph': 7.0,
                'oxygen': 21.0,
                'stress': 2.0
            },
            'acidophile': {
                'temperature': 37.0,
                'ph': 3.0,
                'oxygen': 21.0,
                'stress': 1.0
            }
        }
        
        # 寻找最佳匹配
        for ctx_key, defaults in context_map.items():
            if ctx_key in context:
                return defaults
        
        return {}
    
    def generate_condition_combinations(self, base_conditions: ConditionVector,
                                       variations: Dict[Union[ConditionType, str], List[float]],
                                       max_combinations: int = 100) -> List[ConditionVector]:
        """生成条件组合"""
        combinations = []
        
        # 递归生成所有组合
        def generate_recursive(current_conditions: ConditionVector, 
                             remaining_variations: Dict, depth: int = 0):
            if len(combinations) >= max_combinations:
                return
            
            if not remaining_variations or depth > 10:
                combinations.append(current_conditions.copy())
                return
            
            # 取出一个条件进行变化
            condition_type = next(iter(remaining_variations))
            values = remaining_variations.pop(condition_type)
            
            for value in values:
                new_conditions = current_conditions.copy()
                new_conditions.set_condition(condition_type, value)
                
                # 验证和调整条件
                adjusted_conditions = self.adjust_conditions(new_conditions)
                
                # 递归生成剩余条件的组合
                remaining_copy = remaining_variations.copy()
                generate_recursive(adjusted_conditions, remaining_copy, depth + 1)
        
        generate_recursive(base_conditions, variations.copy())
        
        logger.info(f"生成 {len(combinations)} 个条件组合")
        return combinations
    
    def optimize_conditions(self, target_function: callable, 
                          initial_conditions: ConditionVector,
                          optimization_params: Optional[Dict] = None) -> ConditionVector:
        """条件优化"""
        params = optimization_params or {}
        max_iterations = params.get('max_iterations', 50)
        step_size = params.get('step_size', 0.1)
        
        current_conditions = initial_conditions.copy()
        best_conditions = current_conditions.copy()
        best_score = target_function(current_conditions)
        
        for iteration in range(max_iterations):
            # 生成邻近条件
            candidates = self._generate_neighbors(current_conditions, step_size)
            
            # 评估候选条件
            for candidate in candidates:
                adjusted_candidate = self.adjust_conditions(candidate)
                score = target_function(adjusted_candidate)
                
                if score > best_score:
                    best_score = score
                    best_conditions = adjusted_candidate.copy()
                    current_conditions = adjusted_candidate.copy()
            
            logger.debug(f"优化迭代 {iteration + 1}: 最佳分数 = {best_score:.4f}")
        
        logger.info(f"条件优化完成: 最终分数 = {best_score:.4f}")
        return best_conditions
    
    def _generate_neighbors(self, conditions: ConditionVector, 
                          step_size: float) -> List[ConditionVector]:
        """生成邻近条件"""
        neighbors = []
        
        for condition_key in conditions.conditions.keys():
            current_value = conditions.get_condition(condition_key)
            if current_value is None:
                continue
            
            # 生成增加和减少的邻居
            for direction in [-1, 1]:
                neighbor = conditions.copy()
                new_value = current_value + direction * step_size
                neighbor.set_condition(condition_key, new_value)
                neighbors.append(neighbor)
        
        return neighbors
    
    def save_conditions(self, condition_vector: ConditionVector, 
                       file_path: Union[str, Path]) -> None:
        """保存条件到文件"""
        data = {
            'conditions': condition_vector.conditions,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"条件已保存到: {file_path}")
    
    def load_conditions(self, file_path: Union[str, Path]) -> ConditionVector:
        """从文件加载条件"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        conditions = ConditionVector(data['conditions'])
        logger.info(f"从 {file_path} 加载条件: {len(conditions)} 个")
        
        return conditions
    
    def get_condition_summary(self, condition_vector: ConditionVector) -> Dict[str, Any]:
        """获取条件摘要"""
        normalized = condition_vector.normalize_conditions()
        is_valid, errors = self.validate_conditions(condition_vector)
        
        summary = {
            'total_conditions': len(condition_vector),
            'is_valid': is_valid,
            'errors': errors,
            'normalized_conditions': normalized,
            'raw_conditions': condition_vector.conditions
        }
        
        return summary

def create_condition_controller() -> ConditionController:
    """创建条件控制器的工厂函数"""
    return ConditionController()
