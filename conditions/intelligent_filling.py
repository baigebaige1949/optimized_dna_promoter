#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能条件填充模块
基于生物学知识和机器学习的智能条件填充和组合生成
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from .condition_controller import ConditionVector, ConditionType, ConditionDefinition
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BiologicalKnowledge:
    """生物学知识库"""
    organism_type: str
    optimal_conditions: Dict[str, float]
    tolerance_ranges: Dict[str, Tuple[float, float]]
    metabolic_pathways: List[str]
    stress_responses: Dict[str, float]
    interaction_rules: Dict[str, callable]

class KnowledgeBase:
    """生物学知识库管理器"""
    
    def __init__(self):
        self.organisms = {}
        self.pathway_conditions = {}
        self.interaction_matrices = {}
        self._initialize_default_knowledge()
    
    def _initialize_default_knowledge(self):
        """初始化默认生物学知识"""
        # E. coli 知识
        self.organisms['e_coli'] = BiologicalKnowledge(
            organism_type='prokaryote',
            optimal_conditions={
                'temperature': 37.0,
                'ph': 7.0,
                'oxygen': 21.0,
                'nutrients': 80.0,
                'osmolarity': 300.0
            },
            tolerance_ranges={
                'temperature': (15.0, 45.0),
                'ph': (6.0, 8.0),
                'oxygen': (0.0, 21.0),
                'nutrients': (20.0, 100.0)
            },
            metabolic_pathways=['glycolysis', 'citric_acid_cycle', 'oxidative_phosphorylation'],
            stress_responses={'heat_shock': 42.0, 'cold_shock': 15.0, 'osmotic_shock': 500.0},
            interaction_rules={
                'temp_ph': lambda t, p: 1.0 if 30 <= t <= 42 and 6.5 <= p <= 7.5 else 0.5,
                'oxygen_nutrients': lambda o, n: n * (1.0 + 0.01 * o)
            }
        )
        
        # 酮酒酵母知识
        self.organisms['yeast'] = BiologicalKnowledge(
            organism_type='eukaryote',
            optimal_conditions={
                'temperature': 30.0,
                'ph': 6.0,
                'oxygen': 21.0,
                'nutrients': 60.0,
                'osmolarity': 280.0
            },
            tolerance_ranges={
                'temperature': (10.0, 40.0),
                'ph': (4.5, 8.0),
                'oxygen': (0.0, 21.0),
                'nutrients': (10.0, 90.0)
            },
            metabolic_pathways=['glycolysis', 'fermentation', 'gluconeogenesis'],
            stress_responses={'heat_shock': 37.0, 'ethanol_tolerance': 15.0},
            interaction_rules={
                'temp_fermentation': lambda t: 1.2 if 25 <= t <= 32 else 0.8,
                'ph_growth': lambda p: 1.0 if 5.0 <= p <= 7.0 else max(0.3, 1.0 - 0.2 * abs(p - 6.0))
            }
        )
        
        # 哺乳动物细胞知识
        self.organisms['mammalian'] = BiologicalKnowledge(
            organism_type='eukaryote',
            optimal_conditions={
                'temperature': 37.0,
                'ph': 7.4,
                'oxygen': 5.0,  # CO2培养箱中的低氧环境
                'nutrients': 70.0,
                'osmolarity': 300.0
            },
            tolerance_ranges={
                'temperature': (35.0, 39.0),
                'ph': (7.0, 7.8),
                'oxygen': (1.0, 21.0),
                'nutrients': (30.0, 100.0)
            },
            metabolic_pathways=['glycolysis', 'citric_acid_cycle', 'fatty_acid_oxidation'],
            stress_responses={'hypoxia': 1.0, 'serum_starvation': 10.0},
            interaction_rules={
                'oxygen_metabolism': lambda o: 1.0 if o >= 5.0 else 0.5 + 0.1 * o,
                'ph_buffering': lambda p: max(0.1, 1.0 - 2.0 * abs(p - 7.4))
            }
        )
        
        # 代谢路径与条件关联
        self.pathway_conditions = {
            'glycolysis': {
                'optimal_ph': 7.0,
                'optimal_temp': 37.0,
                'glucose_dependence': True,
                'oxygen_independent': True
            },
            'citric_acid_cycle': {
                'optimal_ph': 7.4,
                'optimal_temp': 37.0,
                'oxygen_dependent': True,
                'mitochondrial': True
            },
            'fermentation': {
                'optimal_ph': 6.0,
                'optimal_temp': 30.0,
                'oxygen_independent': True,
                'anaerobic_preferred': True
            }
        }
        
        logger.info(f"初始化知识库: {len(self.organisms)} 个生物体")
    
    def get_organism_knowledge(self, organism: str) -> Optional[BiologicalKnowledge]:
        """获取生物体知识"""
        return self.organisms.get(organism.lower())
    
    def add_organism_knowledge(self, organism: str, knowledge: BiologicalKnowledge):
        """添加新的生物体知识"""
        self.organisms[organism.lower()] = knowledge
        logger.info(f"添加生物体知识: {organism}")
    
    def get_pathway_requirements(self, pathway: str) -> Optional[Dict]:
        """获取代谢路径需求"""
        return self.pathway_conditions.get(pathway.lower())
    
    def suggest_conditions_for_pathway(self, pathway: str, 
                                      base_conditions: ConditionVector) -> ConditionVector:
        """为特定代谢路径建议条件"""
        requirements = self.get_pathway_requirements(pathway)
        if not requirements:
            return base_conditions
        
        result = base_conditions.copy()
        
        if 'optimal_ph' in requirements:
            result.set_condition('ph', requirements['optimal_ph'])
        
        if 'optimal_temp' in requirements:
            result.set_condition('temperature', requirements['optimal_temp'])
        
        if requirements.get('oxygen_dependent', False):
            result.set_condition('oxygen', 21.0)
        elif requirements.get('anaerobic_preferred', False):
            result.set_condition('oxygen', 0.5)
        
        return result

class ConditionPredictor(nn.Module):
    """条件预测网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # 输出[0,1]范围的标准化值
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class CorrelationAnalyzer:
    """条件相关性分析器"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.feature_names = []
        self.interaction_strength = {}
    
    def analyze_correlations(self, condition_data: List[ConditionVector], 
                           performance_data: Optional[List[float]] = None) -> Dict[str, Any]:
        """分析条件相关性"""
        if not condition_data:
            return {'error': '没有数据'}
        
        # 准备数据
        data_matrix = []
        self.feature_names = sorted(set().union(*[cv.conditions.keys() for cv in condition_data]))
        
        for cv in condition_data:
            row = [cv.get_condition(feature, 0.0) for feature in self.feature_names]
            data_matrix.append(row)
        
        data_array = np.array(data_matrix)
        
        # 计算相关矩阵
        correlation_matrix = np.corrcoef(data_array.T)
        self.correlation_matrix = correlation_matrix
        
        # 分析强相关性
        strong_correlations = []
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                corr = correlation_matrix[i, j]
                if abs(corr) > 0.5:  # 强相关阈值
                    strong_correlations.append({
                        'feature1': self.feature_names[i],
                        'feature2': self.feature_names[j],
                        'correlation': corr,
                        'strength': 'strong' if abs(corr) > 0.7 else 'moderate'
                    })
        
        # 如果有性能数据，分析与性能的相关性
        performance_correlations = []
        if performance_data and len(performance_data) == len(condition_data):
            for i, feature in enumerate(self.feature_names):
                feature_values = data_array[:, i]
                corr, p_value = pearsonr(feature_values, performance_data)
                performance_correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'p_value': p_value,
                    'significance': 'significant' if p_value < 0.05 else 'not_significant'
                })
        
        analysis_result = {
            'correlation_matrix': correlation_matrix.tolist(),
            'feature_names': self.feature_names,
            'strong_correlations': strong_correlations,
            'performance_correlations': performance_correlations,
            'summary': {
                'total_features': len(self.feature_names),
                'strong_correlations_count': len(strong_correlations),
                'data_points': len(condition_data)
            }
        }
        
        return analysis_result
    
    def predict_missing_condition(self, partial_conditions: ConditionVector, 
                                 missing_condition: str) -> Optional[float]:
        """基于相关性预测缺失条件"""
        if self.correlation_matrix is None or missing_condition not in self.feature_names:
            return None
        
        missing_idx = self.feature_names.index(missing_condition)
        
        # 找到与缺失条件相关性最强的已知条件
        best_corr = 0.0
        best_predictor = None
        best_predictor_value = None
        
        for condition, value in partial_conditions.conditions.items():
            if condition in self.feature_names:
                predictor_idx = self.feature_names.index(condition)
                corr = abs(self.correlation_matrix[missing_idx, predictor_idx])
                if corr > best_corr:
                    best_corr = corr
                    best_predictor = condition
                    best_predictor_value = value
        
        if best_predictor and best_corr > 0.3:  # 相关性阈值
            # 简单的线性预测
            predictor_idx = self.feature_names.index(best_predictor)
            correlation = self.correlation_matrix[missing_idx, predictor_idx]
            
            # 获取预测器的默认值范围
            predictor_range = self._get_feature_range(best_predictor)
            missing_range = self._get_feature_range(missing_condition)
            
            if predictor_range and missing_range:
                # 标准化预测器值
                normalized_predictor = (best_predictor_value - predictor_range[0]) / (predictor_range[1] - predictor_range[0])
                
                # 预测标准化的缺失值
                if correlation > 0:
                    normalized_missing = normalized_predictor
                else:
                    normalized_missing = 1.0 - normalized_predictor
                
                # 反标准化
                predicted_value = missing_range[0] + normalized_missing * (missing_range[1] - missing_range[0])
                
                logger.debug(f"基于相关性预测 {missing_condition} = {predicted_value:.2f} (相关性: {correlation:.3f})")
                return predicted_value
        
        return None
    
    def _get_feature_range(self, feature_name: str) -> Optional[Tuple[float, float]]:
        """获取特征的值域范围"""
        # 尝试从条件定义中获取范围
        for condition_type in ConditionType:
            if condition_type.value == feature_name:
                condition_range = ConditionDefinition.get_condition_range(condition_type)
                if condition_range:
                    return (condition_range.min_value, condition_range.max_value)
        
        # 默认范围
        default_ranges = {
            'temperature': (4.0, 85.0),
            'ph': (1.0, 14.0),
            'oxygen': (0.0, 100.0),
            'nutrients': (0.0, 100.0),
            'concentration': (0.001, 1000.0),
            'time': (0.0, 168.0)
        }
        
        return default_ranges.get(feature_name, (0.0, 100.0))

class IntelligentFiller:
    """智能填充器主类"""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.condition_predictor = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 基于规则的填充策略
        self.filling_strategies = [
            self._biological_knowledge_filling,
            self._correlation_based_filling,
            self._ml_based_filling,
            self._default_value_filling
        ]
        
        logger.info("智能填充器初始化完成")
    
    def train_predictor(self, training_data: List[Tuple[ConditionVector, ConditionVector]], 
                       validation_split: float = 0.2) -> Dict[str, float]:
        """训练条件预测器"""
        if not training_data:
            raise ValueError("没有训练数据")
        
        # 准备数据
        input_data = []
        output_data = []
        
        # 获取所有可能的条件类型
        all_conditions = set()
        for partial, complete in training_data:
            all_conditions.update(partial.conditions.keys())
            all_conditions.update(complete.conditions.keys())
        
        condition_list = sorted(all_conditions)
        
        for partial, complete in training_data:
            # 输入：部分条件（缺失的设为-1）
            input_row = []
            for condition in condition_list:
                value = partial.get_condition(condition)
                input_row.append(value if value is not None else -1.0)
            
            # 输出：完整条件
            output_row = []
            for condition in condition_list:
                value = complete.get_condition(condition, 0.0)
                output_row.append(value)
            
            input_data.append(input_row)
            output_data.append(output_row)
        
        # 数据预处理
        input_array = np.array(input_data)
        output_array = np.array(output_data)
        
        # 标准化输出数据
        self.scaler.fit(output_array)
        normalized_output = self.scaler.transform(output_array)
        
        # 分割数据
        split_idx = int(len(input_data) * (1 - validation_split))
        
        X_train = torch.FloatTensor(input_array[:split_idx])
        y_train = torch.FloatTensor(normalized_output[:split_idx])
        X_val = torch.FloatTensor(input_array[split_idx:])
        y_val = torch.FloatTensor(normalized_output[split_idx:])
        
        # 创建模型
        self.condition_predictor = ConditionPredictor(
            input_dim=len(condition_list),
            output_dim=len(condition_list)
        )
        
        # 训练模型
        optimizer = torch.optim.Adam(self.condition_predictor.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(100):
            # 训练
            self.condition_predictor.train()
            optimizer.zero_grad()
            
            train_pred = self.condition_predictor(X_train)
            train_loss = criterion(train_pred, y_train)
            train_loss.backward()
            optimizer.step()
            
            # 验证
            self.condition_predictor.eval()
            with torch.no_grad():
                val_pred = self.condition_predictor(X_val)
                val_loss = criterion(val_pred, y_val)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
        
        self.is_trained = True
        self.condition_order = condition_list
        
        training_result = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'total_epochs': 100,
            'condition_count': len(condition_list)
        }
        
        logger.info(f"模型训练完成: 验证损失 = {val_losses[-1]:.4f}")
        return training_result
    
    def intelligent_fill(self, partial_conditions: ConditionVector, 
                        biological_context: Optional[str] = None,
                        target_pathways: Optional[List[str]] = None) -> ConditionVector:
        """智能填充缺失条件"""
        result = partial_conditions.copy()
        
        # 依次尝试不同的填充策略
        for strategy in self.filling_strategies:
            filled_conditions = strategy(result, biological_context, target_pathways)
            
            # 更新结果，保留原有值
            for condition, value in filled_conditions.conditions.items():
                if result.get_condition(condition) is None:
                    result.set_condition(condition, value)
        
        logger.info(f"填充完成: {len(partial_conditions)} -> {len(result)} 个条件")
        return result
    
    def _biological_knowledge_filling(self, conditions: ConditionVector, 
                                    biological_context: Optional[str] = None,
                                    target_pathways: Optional[List[str]] = None) -> ConditionVector:
        """基于生物学知识的填充"""
        result = conditions.copy()
        
        # 基于生物体知识填充
        if biological_context:
            knowledge = self.knowledge_base.get_organism_knowledge(biological_context)
            if knowledge:
                for condition, value in knowledge.optimal_conditions.items():
                    if result.get_condition(condition) is None:
                        result.set_condition(condition, value)
                        logger.debug(f"基于{biological_context}生物学知识填充 {condition} = {value}")
        
        # 基于代谢路径填充
        if target_pathways:
            for pathway in target_pathways:
                pathway_conditions = self.knowledge_base.suggest_conditions_for_pathway(pathway, result)
                for condition, value in pathway_conditions.conditions.items():
                    if result.get_condition(condition) is None:
                        result.set_condition(condition, value)
                        logger.debug(f"基于{pathway}路径填充 {condition} = {value}")
        
        return result
    
    def _correlation_based_filling(self, conditions: ConditionVector, 
                                 biological_context: Optional[str] = None,
                                 target_pathways: Optional[List[str]] = None) -> ConditionVector:
        """基于相关性分析的填充"""
        result = conditions.copy()
        
        # 找到所有缺失的条件
        all_possible_conditions = [ct.value for ct in ConditionType if ct != ConditionType.CUSTOM]
        missing_conditions = [c for c in all_possible_conditions 
                            if result.get_condition(c) is None]
        
        # 尝试预测缺失条件
        for missing_condition in missing_conditions:
            predicted_value = self.correlation_analyzer.predict_missing_condition(result, missing_condition)
            if predicted_value is not None:
                result.set_condition(missing_condition, predicted_value)
                logger.debug(f"基于相关性填充 {missing_condition} = {predicted_value:.2f}")
        
        return result
    
    def _ml_based_filling(self, conditions: ConditionVector, 
                         biological_context: Optional[str] = None,
                         target_pathways: Optional[List[str]] = None) -> ConditionVector:
        """基于机器学习的填充"""
        if not self.is_trained or self.condition_predictor is None:
            return conditions
        
        result = conditions.copy()
        
        # 准备输入数据
        input_row = []
        for condition in self.condition_order:
            value = result.get_condition(condition)
            input_row.append(value if value is not None else -1.0)
        
        # 预测
        self.condition_predictor.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor([input_row])
            predicted_normalized = self.condition_predictor(input_tensor)
            predicted_values = self.scaler.inverse_transform(predicted_normalized.numpy())[0]
        
        # 填充缺失值
        for i, condition in enumerate(self.condition_order):
            if result.get_condition(condition) is None:
                result.set_condition(condition, predicted_values[i])
                logger.debug(f"基于机器学习填充 {condition} = {predicted_values[i]:.2f}")
        
        return result
    
    def _default_value_filling(self, conditions: ConditionVector, 
                              biological_context: Optional[str] = None,
                              target_pathways: Optional[List[str]] = None) -> ConditionVector:
        """默认值填充（最后的后备策略）"""
        result = conditions.copy()
        
        # 使用预定义条件的默认值
        for condition_type, condition_range in ConditionDefinition.PREDEFINED_CONDITIONS.items():
            if isinstance(condition_type, ConditionType):
                condition_name = condition_type.value
                if result.get_condition(condition_name) is None:
                    result.set_condition(condition_name, condition_range.default_value)
                    logger.debug(f"默认值填充 {condition_name} = {condition_range.default_value}")
        
        return result
    
    def batch_fill(self, partial_conditions_list: List[ConditionVector],
                   biological_contexts: Optional[List[str]] = None,
                   target_pathways_list: Optional[List[List[str]]] = None) -> List[ConditionVector]:
        """批量填充条件"""
        results = []
        
        for i, partial_conditions in enumerate(partial_conditions_list):
            context = biological_contexts[i] if biological_contexts and i < len(biological_contexts) else None
            pathways = target_pathways_list[i] if target_pathways_list and i < len(target_pathways_list) else None
            
            filled_conditions = self.intelligent_fill(partial_conditions, context, pathways)
            results.append(filled_conditions)
        
        logger.info(f"批量填充完成: {len(results)} 个条件集")
        return results
    
    def validate_filled_conditions(self, filled_conditions: ConditionVector,
                                  biological_context: Optional[str] = None) -> Dict[str, Any]:
        """验证填充的条件"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'biological_compatibility': True,
            'missing_critical_conditions': []
        }
        
        # 检查条件范围
        for condition_type in ConditionType:
            if condition_type == ConditionType.CUSTOM:
                continue
            
            condition_name = condition_type.value
            value = filled_conditions.get_condition(condition_name)
            
            if value is not None:
                condition_range = ConditionDefinition.get_condition_range(condition_type)
                if condition_range and not condition_range.is_valid(value):
                    validation_result['warnings'].append(
                        f"{condition_name} 值 {value} 超出推荐范围 [{condition_range.min_value}, {condition_range.max_value}]"
                    )
        
        # 检查关键条件
        critical_conditions = ['temperature', 'ph']
        for critical in critical_conditions:
            if filled_conditions.get_condition(critical) is None:
                validation_result['missing_critical_conditions'].append(critical)
        
        # 检查生物学兼容性
        if biological_context:
            knowledge = self.knowledge_base.get_organism_knowledge(biological_context)
            if knowledge:
                for condition, value in filled_conditions.conditions.items():
                    if condition in knowledge.tolerance_ranges:
                        min_tol, max_tol = knowledge.tolerance_ranges[condition]
                        if not (min_tol <= value <= max_tol):
                            validation_result['biological_compatibility'] = False
                            validation_result['errors'].append(
                                f"{condition} 值 {value} 超出{biological_context}的耐受范围 [{min_tol}, {max_tol}]"
                            )
        
        # 综合验证结果
        validation_result['is_valid'] = (len(validation_result['errors']) == 0 and 
                                       len(validation_result['missing_critical_conditions']) == 0)
        
        return validation_result
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        save_data = {
            'model_state_dict': self.condition_predictor.state_dict(),
            'scaler': self.scaler,
            'condition_order': self.condition_order,
            'is_trained': self.is_trained
        }
        
        torch.save(save_data, file_path)
        logger.info(f"模型已保存到: {file_path}")
    
    def load_model(self, file_path: Union[str, Path]) -> None:
        """加载模型"""
        save_data = torch.load(file_path, map_location='cpu')
        
        self.condition_order = save_data['condition_order']
        self.condition_predictor = ConditionPredictor(
            input_dim=len(self.condition_order),
            output_dim=len(self.condition_order)
        )
        self.condition_predictor.load_state_dict(save_data['model_state_dict'])
        self.scaler = save_data['scaler']
        self.is_trained = save_data['is_trained']
        
        logger.info(f"模型已从 {file_path} 加载")
    
    def get_filling_statistics(self) -> Dict[str, Any]:
        """获取填充统计信息"""
        stats = {
            'knowledge_base': {
                'organisms_count': len(self.knowledge_base.organisms),
                'pathways_count': len(self.knowledge_base.pathway_conditions),
                'available_organisms': list(self.knowledge_base.organisms.keys())
            },
            'predictor': {
                'is_trained': self.is_trained,
                'condition_count': len(self.condition_order) if hasattr(self, 'condition_order') else 0
            },
            'strategies_count': len(self.filling_strategies)
        }
        
        return stats

def create_intelligent_filler() -> IntelligentFiller:
    """创建智能填充器的工厂函数"""
    return IntelligentFiller()
