"""条件扩散模型实现

基于DNA-Diffusion论文实现的条件U-Net架构，支持多种条件输入：
- 细胞类型
- 环境条件（温度、pH、氧气含量）
- 生长周期
- 其他生物学条件

主要特性：
1. 灵活的条件控制接口
2. 条件嵌入层和交叉注意力机制
3. 无分类器引导(Classifier-free Guidance)
4. 与现有diffusion_model.py的兼容性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from ..config.model_config import DiffusionModelConfig
from ..utils.logger import get_logger
from .diffusion_model import (
    NoiseScheduler, SinusoidalPositionEmbedding, 
    SelfAttention1D, ResidualBlock
)

logger = get_logger(__name__)


class ConditionConfig:
    """条件配置类，定义所有可用的条件类型"""
    
    # 细胞类型条件
    CELL_TYPES = ['E.coli', 'Yeast', 'HEK293', 'CHO', 'Other']
    
    # 环境条件范围
    TEMPERATURE_RANGE = (25, 42)      # 摄氏度
    PH_RANGE = (5.0, 8.5)            # pH值
    OXYGEN_RANGE = (0, 21)           # 氧气含量百分比
    
    # 生长周期条件
    GROWTH_PHASES = ['Lag', 'Log', 'Stationary', 'Death']
    
    # 默认值
    DEFAULT_VALUES = {
        'cell_type': 'E.coli',
        'temperature': 37.0,
        'ph': 7.0,
        'oxygen': 21.0,
        'growth_phase': 'Log',
        'nutrient_level': 1.0,
        'stress_level': 0.0
    }
    
    @classmethod
    def get_condition_dims(cls) -> Dict[str, int]:
        """返回每个条件的维度"""
        return {
            'cell_type': len(cls.CELL_TYPES),
            'temperature': 1,
            'ph': 1,
            'oxygen': 1,
            'growth_phase': len(cls.GROWTH_PHASES),
            'nutrient_level': 1,
            'stress_level': 1
        }
    
    @classmethod
    def get_total_condition_dim(cls) -> int:
        """返回总条件维度"""
        return sum(cls.get_condition_dims().values())


class ConditionEmbedding(nn.Module):
    """条件嵌入层，将各种条件编码为统一的嵌入向量"""
    
    def __init__(self, condition_dim: int, embedding_dim: int):
        super().__init__()
        self.condition_dim = condition_dim
        self.embedding_dim = embedding_dim
        
        # 细胞类型嵌入（分类）
        self.cell_type_embedding = nn.Embedding(len(ConditionConfig.CELL_TYPES), 64)
        
        # 生长周期嵌入（分类）
        self.growth_phase_embedding = nn.Embedding(len(ConditionConfig.GROWTH_PHASES), 32)
        
        # 连续值条件的线性投影
        self.continuous_projection = nn.Sequential(
            nn.Linear(4, 64),  # temperature, ph, oxygen, nutrient_level, stress_level
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        
        # 最终投影到目标维度
        total_concat_dim = 64 + 32 + 64  # cell_type + growth_phase + continuous
        self.final_projection = nn.Sequential(
            nn.Linear(total_concat_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            conditions: 包含各种条件的字典
                - cell_type: [batch_size] 细胞类型索引
                - growth_phase: [batch_size] 生长周期索引
                - temperature: [batch_size] 温度
                - ph: [batch_size] pH值
                - oxygen: [batch_size] 氧气含量
                - nutrient_level: [batch_size] 营养水平
                - stress_level: [batch_size] 应力水平
        
        Returns:
            torch.Tensor: [batch_size, embedding_dim] 条件嵌入
        """
        batch_size = next(iter(conditions.values())).size(0)
        device = next(iter(conditions.values())).device
        
        embeddings = []
        
        # 细胞类型嵌入
        if 'cell_type' in conditions:
            cell_emb = self.cell_type_embedding(conditions['cell_type'])
        else:
            # 使用默认细胞类型
            default_cell = torch.zeros(batch_size, dtype=torch.long, device=device)
            cell_emb = self.cell_type_embedding(default_cell)
        embeddings.append(cell_emb)
        
        # 生长周期嵌入
        if 'growth_phase' in conditions:
            phase_emb = self.growth_phase_embedding(conditions['growth_phase'])
        else:
            # 使用默认生长周期
            default_phase = torch.ones(batch_size, dtype=torch.long, device=device)  # Log phase
            phase_emb = self.growth_phase_embedding(default_phase)
        embeddings.append(phase_emb)
        
        # 连续值条件
        continuous_values = []
        for key in ['temperature', 'ph', 'oxygen', 'nutrient_level']:
            if key in conditions:
                values = conditions[key]
            else:
                # 使用默认值
                default_val = ConditionConfig.DEFAULT_VALUES.get(key, 0.0)
                values = torch.full((batch_size,), default_val, device=device)
            continuous_values.append(values.unsqueeze(-1))
        
        continuous_tensor = torch.cat(continuous_values, dim=-1)
        continuous_emb = self.continuous_projection(continuous_tensor)
        embeddings.append(continuous_emb)
        
        # 拼接所有嵌入
        combined_emb = torch.cat(embeddings, dim=-1)
        
        # 最终投影
        final_emb = self.final_projection(combined_emb)
        final_emb = self.dropout(final_emb)
        
        return final_emb


class CrossAttention1D(nn.Module):
    """一维交叉注意力机制，用于条件和序列特征的交互"""
    
    def __init__(self, query_dim: int, key_value_dim: int, num_heads: int = 8):
        super().__init__()
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        self.norm_query = nn.GroupNorm(min(8, query_dim), query_dim)
        self.norm_kv = nn.LayerNorm(key_value_dim)
        
        self.query_proj = nn.Conv1d(query_dim, query_dim, kernel_size=1)
        self.key_proj = nn.Linear(key_value_dim, query_dim)
        self.value_proj = nn.Linear(key_value_dim, query_dim)
        self.out_proj = nn.Conv1d(query_dim, query_dim, kernel_size=1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch_size, channels, length] 序列特征
            key_value: [batch_size, key_value_dim] 条件嵌入
        
        Returns:
            torch.Tensor: [batch_size, channels, length] 注意力后的序列特征
        """
        b, c, l = query.shape
        
        # 归一化
        q = self.norm_query(query)
        kv = self.norm_kv(key_value)
        
        # 投影
        q = self.query_proj(q)  # [b, c, l]
        k = self.key_proj(kv)   # [b, c]
        v = self.value_proj(kv) # [b, c]
        
        # 重塑为多头注意力格式
        q = q.reshape(b, self.num_heads, self.head_dim, l).permute(0, 1, 3, 2)  # [b, h, l, d]
        k = k.reshape(b, self.num_heads, self.head_dim).unsqueeze(2)            # [b, h, 1, d]
        v = v.reshape(b, self.num_heads, self.head_dim).unsqueeze(2)            # [b, h, 1, d]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [b, h, l, 1]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_out = torch.matmul(attn_weights, v)  # [b, h, l, d]
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(b, c, l)  # [b, c, l]
        
        # 输出投影
        out = self.out_proj(attn_out)
        
        # 残差连接
        return query + out


class ConditionalResidualBlock(nn.Module):
    """带条件控制的残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 condition_emb_dim: int, dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 条件控制的投影层
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_emb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) 
            if in_channels != out_channels 
            else nn.Identity()
        )
        
        # 交叉注意力机制
        if use_attention:
            self.self_attention = SelfAttention1D(out_channels)
            self.cross_attention = CrossAttention1D(out_channels, condition_emb_dim)
        else:
            self.self_attention = None
            self.cross_attention = None
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, 
                condition_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, length] 输入特征
            time_emb: [batch_size, time_emb_dim] 时间嵌入
            condition_emb: [batch_size, condition_emb_dim] 条件嵌入
        """
        h = self.block1(x)
        
        # 添加时间嵌入
        time_emb_proj = self.time_mlp(time_emb)[:, :, None]
        h = h + time_emb_proj
        
        # 添加条件嵌入
        condition_emb_proj = self.condition_mlp(condition_emb)[:, :, None]
        h = h + condition_emb_proj
        
        h = self.block2(h)
        
        # 残差连接
        h = h + self.shortcut(x)
        
        # 注意力机制
        if self.self_attention is not None:
            h = self.self_attention(h)
        if self.cross_attention is not None:
            h = self.cross_attention(h, condition_emb)
        
        return h


class ConditionalUNet1D(nn.Module):
    """条件控制的一维U-Net模型"""
    
    def __init__(self, config: DiffusionModelConfig, in_channels: int = 4, 
                 out_channels: int = 4, condition_emb_dim: int = 256):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_emb_dim = condition_emb_dim
        
        # 时间嵌入
        time_emb_dim = config.hidden_dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 条件嵌入
        total_condition_dim = ConditionConfig.get_total_condition_dim()
        self.condition_embedding = ConditionEmbedding(total_condition_dim, condition_emb_dim)
        
        # 编码器
        self.input_conv = nn.Conv1d(in_channels, config.unet_channels[0], kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        channels = config.unet_channels
        for i in range(len(channels) - 1):
            self.down_blocks.append(ConditionalResidualBlock(
                channels[i], channels[i+1], time_emb_dim, condition_emb_dim,
                config.dropout, use_attention=(channels[i+1] >= 128)
            ))
        
        # 中间层
        mid_channels = channels[-1]
        self.mid_block1 = ConditionalResidualBlock(
            mid_channels, mid_channels, time_emb_dim, condition_emb_dim, 
            config.dropout, use_attention=True
        )
        self.mid_block2 = ConditionalResidualBlock(
            mid_channels, mid_channels, time_emb_dim, condition_emb_dim, 
            config.dropout, use_attention=True
        )
        
        # 解码器
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(ConditionalResidualBlock(
                channels[i] + channels[i-1], channels[i-1], time_emb_dim, condition_emb_dim,
                config.dropout, use_attention=(channels[i-1] >= 128)
            ))
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1)
        )
        
        # 无分类器引导的空条件嵌入
        self.null_condition_emb = nn.Parameter(torch.randn(condition_emb_dim))
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                conditions: Optional[Dict[str, torch.Tensor]] = None,
                use_cfg: bool = False, cfg_drop_prob: float = 0.1) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, length] 输入序列
            timesteps: [batch_size] 时间步
            conditions: 条件字典
            use_cfg: 是否使用无分类器引导训练
            cfg_drop_prob: 条件丢弃概率
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)
        
        # 条件嵌入
        if conditions is None:
            conditions = {}
        
        # 无分类器引导：训练时随机丢弃条件
        if use_cfg and self.training:
            mask = torch.rand(batch_size, device=device) < cfg_drop_prob
            condition_emb = self.condition_embedding(conditions)
            null_emb = self.null_condition_emb.unsqueeze(0).expand(batch_size, -1)
            condition_emb = torch.where(mask.unsqueeze(-1), null_emb, condition_emb)
        else:
            condition_emb = self.condition_embedding(conditions)
        
        # U-Net前向传播
        h = self.input_conv(x)
        
        # 编码器
        skip_connections = [h]
        for block in self.down_blocks:
            h = block(h, time_emb, condition_emb)
            skip_connections.append(h)
            h = F.avg_pool1d(h, kernel_size=2)
        
        # 中间层
        h = self.mid_block1(h, time_emb, condition_emb)
        h = self.mid_block2(h, time_emb, condition_emb)
        
        # 解码器
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, skip_connections.pop()], dim=1)
            h = block(h, time_emb, condition_emb)
        
        # 输出层
        return self.output_conv(h)


class ConditionalDiffusionModel(nn.Module):
    """主条件扩散模型"""
    
    def __init__(self, config: DiffusionModelConfig, vocab_size: int = 4,
                 condition_emb_dim: int = 256):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.condition_emb_dim = condition_emb_dim
        
        # 噪声调度器
        self.noise_scheduler = NoiseScheduler(config)
        
        # 条件U-Net模型
        self.unet = ConditionalUNet1D(config, vocab_size, vocab_size, condition_emb_dim)
        
        # 条件验证器
        self.condition_validator = ConditionValidator()
        
        logger.info(f"Initialized conditional diffusion model with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor = None, 
                conditions: Optional[Dict[str, torch.Tensor]] = None,
                use_cfg: bool = False, cfg_drop_prob: float = 0.1) -> torch.Tensor:
        """前向传播"""
        if timesteps is None:
            timesteps = self.noise_scheduler.sample_timesteps(x.shape[0], x.device)
        
        return self.unet(x, timesteps, conditions, use_cfg, cfg_drop_prob)
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """训练步骤"""
        x_0 = batch['sequences']
        conditions = {k: v for k, v in batch.items() if k != 'sequences'}
        
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 采样时间步
        timesteps = self.noise_scheduler.sample_timesteps(batch_size, device)
        
        # 添加噪声
        noise = torch.randn_like(x_0)
        x_t, _ = self.noise_scheduler.add_noise(x_0, timesteps, noise)
        
        # 预测噪声
        predicted_noise = self.forward(x_t, timesteps, conditions, 
                                     use_cfg=True, cfg_drop_prob=0.1)
        
        # 计算损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return {
            'loss': loss,
            'timesteps': timesteps,
            'noise_mse': loss
        }
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], conditions: Optional[Dict[str, torch.Tensor]] = None,
               num_inference_steps: int = 50, guidance_scale: float = 7.5,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """采样生成序列
        
        Args:
            shape: 生成序列的形状
            conditions: 条件字典
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            generator: 随机数生成器
            
        Returns:
            生成的序列
        """
        device = next(self.parameters()).device
        batch_size = shape[0]
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device, generator=generator)
        
        # 验证条件
        if conditions is not None:
            conditions = self.condition_validator.validate_and_complete(conditions, batch_size)
        
        # DDPM采样
        timesteps = torch.linspace(self.config.num_timesteps-1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            if guidance_scale > 1.0 and conditions is not None:
                # 无分类器引导
                # 条件预测
                noise_pred_cond = self.unet(x, t_batch, conditions)
                
                # 无条件预测
                noise_pred_uncond = self.unet(x, t_batch, None)
                
                # 引导
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(x, t_batch, conditions)
            
            # DDPM更新
            alpha_t = self.noise_scheduler.alphas_cumprod[t]
            alpha_t_prev = self.noise_scheduler.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0)
            
            beta_t = 1 - alpha_t / alpha_t_prev
            x = (1 / torch.sqrt(alpha_t / alpha_t_prev)) * (x - beta_t / torch.sqrt(1 - alpha_t) * noise_pred)
            
            if t > 0:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x, generator=generator)
        
        return x
    
    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.noise_scheduler.to(device)
        return self


class ConditionValidator:
    """条件验证和处理器"""
    
    def __init__(self):
        self.condition_config = ConditionConfig()
    
    def validate_and_complete(self, conditions: Dict[str, torch.Tensor], 
                            batch_size: int) -> Dict[str, torch.Tensor]:
        """验证并补全条件"""
        validated_conditions = {}
        device = next(iter(conditions.values())).device if conditions else torch.device('cpu')
        
        for condition_name, default_value in self.condition_config.DEFAULT_VALUES.items():
            if condition_name in conditions:
                validated_conditions[condition_name] = conditions[condition_name]
            else:
                # 使用默认值
                if condition_name in ['cell_type', 'growth_phase']:
                    # 分类变量
                    if condition_name == 'cell_type':
                        default_idx = 0  # E.coli
                    else:  # growth_phase
                        default_idx = 1  # Log
                    validated_conditions[condition_name] = torch.full(
                        (batch_size,), default_idx, dtype=torch.long, device=device
                    )
                else:
                    # 连续变量
                    validated_conditions[condition_name] = torch.full(
                        (batch_size,), default_value, dtype=torch.float, device=device
                    )
        
        return validated_conditions
    
    def encode_cell_type(self, cell_type: str) -> int:
        """编码细胞类型为索引"""
        if cell_type in self.condition_config.CELL_TYPES:
            return self.condition_config.CELL_TYPES.index(cell_type)
        return len(self.condition_config.CELL_TYPES) - 1  # Other
    
    def encode_growth_phase(self, growth_phase: str) -> int:
        """编码生长周期为索引"""
        if growth_phase in self.condition_config.GROWTH_PHASES:
            return self.condition_config.GROWTH_PHASES.index(growth_phase)
        return 1  # Log phase as default


def create_conditional_diffusion_model(config: DiffusionModelConfig, 
                                     vocab_size: int = 4,
                                     condition_emb_dim: int = 256) -> ConditionalDiffusionModel:
    """创建条件扩散模型的工厂函数
    
    Args:
        config: 扩散模型配置
        vocab_size: 词汇表大小（DNA: A,T,C,G = 4）
        condition_emb_dim: 条件嵌入维度
        
    Returns:
        ConditionalDiffusionModel: 条件扩散模型实例
        
    Example:
        >>> from optimized_dna_promoter.config.model_config import DiffusionModelConfig
        >>> config = DiffusionModelConfig()
        >>> model = create_conditional_diffusion_model(config)
        >>> 
        >>> # 准备条件
        >>> conditions = {
        ...     'cell_type': torch.tensor([0]),  # E.coli
        ...     'temperature': torch.tensor([37.0]),
        ...     'ph': torch.tensor([7.0])
        ... }
        >>> 
        >>> # 生成序列
        >>> generated_sequences = model.sample(
        ...     shape=(1, 4, 1000), 
        ...     conditions=conditions,
        ...     guidance_scale=7.5
        ... )
    """
    model = ConditionalDiffusionModel(config, vocab_size, condition_emb_dim)
    logger.info("Created conditional diffusion model with factory function")
    return model


# 导出主要类和函数
__all__ = [
    'ConditionalDiffusionModel',
    'ConditionConfig',
    'ConditionEmbedding',
    'ConditionalUNet1D',
    'CrossAttention1D',
    'ConditionValidator',
    'create_conditional_diffusion_model'
]