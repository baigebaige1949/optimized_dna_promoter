"""
Dirichlet Diffusion Schrödinger Bridge (DDSB) for DNA序列生成
基于DDSM论文实现的Jacobi扩散过程，用于处理离散DNA序列的扩散生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from functools import partial
import math

class StickBreakingTransform:
    """
    Stick-breaking构造，用于将4维DNA序列映射到概率单纯形空间
    处理A, T, G, C四种碱基的概率分布
    """
    
    @staticmethod
    def stick_breaking_to_simplex(stick_breaking: torch.Tensor) -> torch.Tensor:
        """
        将stick-breaking参数转换为单纯形上的概率分布
        
        Args:
            stick_breaking: (..., 3) 维度的参数，用于4维DNA序列
        
        Returns:
            simplex: (..., 4) 维度的概率分布，表示A,T,G,C的概率
        """
        # 使用sigmoid激活确保参数在(0,1)范围内
        betas = torch.sigmoid(stick_breaking)
        
        # 计算累积剩余概率
        remaining_prob = torch.ones_like(betas[..., :1])  # 初始剩余概率为1
        
        # 构建单纯形概率分布
        probs = []
        for i in range(betas.shape[-1]):
            current_prob = remaining_prob * betas[..., i:i+1]
            probs.append(current_prob)
            remaining_prob = remaining_prob * (1 - betas[..., i:i+1])
        
        # 最后一个概率是剩余的所有概率
        probs.append(remaining_prob)
        
        return torch.cat(probs, dim=-1)
    
    @staticmethod
    def simplex_to_stick_breaking(simplex: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        将单纯形概率分布转换回stick-breaking参数
        
        Args:
            simplex: (..., 4) 维度的概率分布
            eps: 数值稳定性参数
        
        Returns:
            stick_breaking: (..., 3) 维度的参数
        """
        # 确保概率非负且和为1
        simplex = torch.clamp(simplex, min=eps)
        simplex = simplex / (simplex.sum(dim=-1, keepdim=True) + eps)
        
        # 逆向计算stick-breaking参数
        remaining_prob = torch.ones_like(simplex[..., :1])
        betas = []
        
        for i in range(simplex.shape[-1] - 1):
            beta = simplex[..., i:i+1] / (remaining_prob + eps)
            beta = torch.clamp(beta, min=eps, max=1-eps)
            betas.append(beta)
            remaining_prob = remaining_prob - simplex[..., i:i+1]
            remaining_prob = torch.clamp(remaining_prob, min=eps)
        
        stick_breaking = torch.cat(betas, dim=-1)
        # 使用logit逆变换
        return torch.log(stick_breaking / (1 - stick_breaking + eps) + eps)


class JacobiProcess:
    """
    Jacobi扩散过程实现，用于在概率单纯形空间进行扩散
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        """
        初始化Jacobi过程
        
        Args:
            alpha, beta: Jacobi过程的参数，控制扩散特性
        """
        self.alpha = alpha
        self.beta = beta
    
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算Jacobi过程的漂移项
        
        Args:
            x: 当前状态 (..., K-1) 维度的stick-breaking参数
            t: 时间 (..., 1)
        
        Returns:
            drift: 漂移向量
        """
        # Jacobi过程的漂移项
        # d_t = (alpha - 1) * (1 - x) / x - (beta - 1) * x / (1 - x)
        eps = 1e-8
        x = torch.clamp(torch.sigmoid(x), min=eps, max=1-eps)
        
        drift = (self.alpha - 1) / (x + eps) - (self.alpha + self.beta - 2) / (1 + eps)
        return drift
    
    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算Jacobi过程的扩散系数
        
        Args:
            x: 当前状态
            t: 时间
        
        Returns:
            diffusion: 扩散系数
        """
        eps = 1e-8
        x = torch.clamp(torch.sigmoid(x), min=eps, max=1-eps)
        
        # Jacobi过程的扩散系数: sqrt(x * (1 - x))
        diffusion = torch.sqrt(x * (1 - x) + eps)
        return diffusion
    
    def sample_prior(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        从Jacobi分布的先验分布采样
        
        Args:
            shape: 采样形状
            device: 设备
        
        Returns:
            samples: 先验采样
        """
        # 从Beta分布采样，然后转换为stick-breaking格式
        beta_samples = torch.distributions.Beta(self.alpha, self.beta).sample(shape).to(device)
        
        # 转换为logit空间（stick-breaking参数）
        eps = 1e-8
        beta_samples = torch.clamp(beta_samples, min=eps, max=1-eps)
        return torch.log(beta_samples / (1 - beta_samples))


class TimeDilation:
    """
    时间膨胀技术，用于提高生成质量
    """
    
    def __init__(self, dilation_factor: float = 2.0):
        """
        Args:
            dilation_factor: 膨胀因子，控制时间变换的强度
        """
        self.dilation_factor = dilation_factor
    
    def forward_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        正向时间变换
        
        Args:
            t: 原始时间 [0, 1]
        
        Returns:
            dilated_t: 膨胀后的时间
        """
        # 使用指数变换进行时间膨胀
        return 1.0 - torch.exp(-self.dilation_factor * t)
    
    def inverse_time(self, dilated_t: torch.Tensor) -> torch.Tensor:
        """
        逆向时间变换
        
        Args:
            dilated_t: 膨胀后的时间
        
        Returns:
            t: 原始时间
        """
        eps = 1e-8
        return -torch.log(torch.clamp(1.0 - dilated_t, min=eps)) / self.dilation_factor
    
    def time_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算时间变换的导数
        
        Args:
            t: 原始时间
        
        Returns:
            dt_ddilated_t: 时间导数
        """
        return self.dilation_factor * torch.exp(-self.dilation_factor * t)


class VariationalScoreMatching:
    """
    变分不变的得分匹配损失函数
    """
    
    @staticmethod
    def score_matching_loss(
        score_network: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        true_score: torch.Tensor,
        importance_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算变分得分匹配损失
        
        Args:
            score_network: 得分网络
            x_t: 时间t的状态
            t: 时间步
            true_score: 真实得分
            importance_weight: 重要性权重用于减少方差
        
        Returns:
            loss: 得分匹配损失
        """
        # 计算预测得分
        pred_score = score_network(x_t, t)
        
        # 计算得分匹配损失
        score_diff = pred_score - true_score
        loss = 0.5 * torch.sum(score_diff ** 2, dim=-1)
        
        # 应用重要性权重
        if importance_weight is not None:
            loss = loss * importance_weight
        
        return torch.mean(loss)
    
    @staticmethod
    def compute_importance_weights(
        x_t: torch.Tensor,
        t: torch.Tensor,
        jacobi_process: JacobiProcess
    ) -> torch.Tensor:
        """
        计算重要性采样权重以减少训练方差
        
        Args:
            x_t: 当前状态
            t: 时间
            jacobi_process: Jacobi过程
        
        Returns:
            weights: 重要性权重
        """
        # 基于扩散系数计算权重
        diffusion_coeff = jacobi_process.diffusion(x_t, t)
        
        # 权重与扩散强度成反比，在扩散较弱的区域给予更高权重
        weights = 1.0 / (torch.sum(diffusion_coeff ** 2, dim=-1) + 1e-8)
        
        return weights


class DirichletDiffusionModel(nn.Module):
    """
    主要的Dirichlet扩散模型，整合所有组件
    """
    
    def __init__(
        self,
        sequence_length: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        alpha: float = 2.0,
        beta: float = 2.0,
        dilation_factor: float = 2.0
    ):
        """
        初始化Dirichlet扩散模型
        
        Args:
            sequence_length: DNA序列长度
            hidden_dim: 隐藏维度
            num_layers: transformer层数
            num_heads: 注意力头数
            alpha, beta: Jacobi过程参数
            dilation_factor: 时间膨胀因子
        """
        super().__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # 核心组件
        self.stick_breaking = StickBreakingTransform()
        self.jacobi_process = JacobiProcess(alpha, beta)
        self.time_dilation = TimeDilation(dilation_factor)
        self.score_matching = VariationalScoreMatching()
        
        # 时间编码
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length, hidden_dim))
        
        # 输入投影 (3维stick-breaking参数)
        self.input_projection = nn.Linear(3, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层（预测得分）
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # 输出3维stick-breaking得分
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算得分函数
        
        Args:
            x_t: 当前状态 (batch_size, seq_len, 3) stick-breaking参数
            t: 时间步 (batch_size, 1)
        
        Returns:
            score: 预测的得分函数 (batch_size, seq_len, 3)
        """
        batch_size, seq_len, _ = x_t.shape
        
        # 时间嵌入
        t_emb = self.time_embedding(t)  # (batch_size, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        
        # 输入投影
        x_emb = self.input_projection(x_t)  # (batch_size, seq_len, hidden_dim)
        
        # 添加位置编码和时间编码
        x_emb = x_emb + self.pos_embedding + t_emb
        
        # Transformer处理
        x_out = self.transformer(x_emb)  # (batch_size, seq_len, hidden_dim)
        
        # 输出得分
        score = self.output_projection(x_out)  # (batch_size, seq_len, 3)
        
        return score
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失
        
        Args:
            x_0: 真实DNA序列的one-hot编码 (batch_size, seq_len, 4)
            noise: 可选的噪声，用于确定性训练
        
        Returns:
            losses: 损失字典
        """
        batch_size, seq_len, _ = x_0.shape
        device = x_0.device
        
        # 转换为stick-breaking表示
        x_0_sb = self.stick_breaking.simplex_to_stick_breaking(x_0)
        
        # 随机采样时间步
        t = torch.rand(batch_size, 1, device=device)
        dilated_t = self.time_dilation.forward_time(t)
        
        # 从先验分布采样噪声
        if noise is None:
            x_T = self.jacobi_process.sample_prior((batch_size, seq_len, 3), device)
        else:
            x_T = noise
        
        # 根据Jacobi过程插值得到x_t
        # 这是简化的线性插值，实际应该使用Jacobi过程的精确解
        sqrt_alpha_t = torch.sqrt(dilated_t).unsqueeze(-1)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - dilated_t).unsqueeze(-1)
        
        x_t = sqrt_alpha_t * x_0_sb + sqrt_one_minus_alpha_t * x_T
        
        # 计算真实得分（近似）
        true_score = -(x_t - x_0_sb) / (1.0 - dilated_t + 1e-8).unsqueeze(-1)
        
        # 预测得分
        pred_score = self.forward(x_t, dilated_t)
        
        # 计算重要性权重
        importance_weights = self.score_matching.compute_importance_weights(
            x_t, dilated_t, self.jacobi_process
        )
        
        # 得分匹配损失
        score_loss = self.score_matching.score_matching_loss(
            lambda x, t: self.forward(x, t),
            x_t, dilated_t, true_score, importance_weights
        )
        
        return {
            'score_loss': score_loss,
            'total_loss': score_loss
        }
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        num_steps: int = 100,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        生成DNA序列样本
        
        Args:
            batch_size: 批次大小
            sequence_length: 序列长度
            num_steps: 采样步数
            temperature: 采样温度
        
        Returns:
            samples: 生成的DNA序列one-hot编码 (batch_size, seq_len, 4)
        """
        device = next(self.parameters()).device
        
        # 从先验分布开始
        x_t = self.jacobi_process.sample_prior((batch_size, sequence_length, 3), device)
        
        # 逐步去噪
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size, 1), 1.0 - i * dt, device=device)
            dilated_t = self.time_dilation.forward_time(t)
            
            # 预测得分
            score = self.forward(x_t, dilated_t)
            
            # 计算漂移项
            drift = self.jacobi_process.drift(x_t, dilated_t)
            
            # 计算扩散项
            diffusion = self.jacobi_process.diffusion(x_t, dilated_t)
            
            # Euler-Maruyama更新
            if i < num_steps - 1:  # 最后一步不加噪声
                dW = torch.randn_like(x_t) * math.sqrt(dt) * temperature
                x_t = x_t + (drift + score) * dt + diffusion.unsqueeze(-1) * dW
            else:
                x_t = x_t + (drift + score) * dt
        
        # 转换回概率单纯形
        simplex_probs = self.stick_breaking.stick_breaking_to_simplex(x_t)
        
        return simplex_probs
    
    @torch.no_grad()
    def compute_likelihood(self, x_0: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """
        计算给定序列的似然
        
        Args:
            x_0: DNA序列 (batch_size, seq_len, 4)
            num_steps: 似然估计步数
        
        Returns:
            log_likelihood: 对数似然 (batch_size,)
        """
        batch_size, seq_len, _ = x_0.shape
        device = x_0.device
        
        # 转换为stick-breaking表示
        x_0_sb = self.stick_breaking.simplex_to_stick_breaking(x_0)
        
        log_likelihood = torch.zeros(batch_size, device=device)
        
        # 使用重要性采样估计似然
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=device)
            dilated_t = self.time_dilation.forward_time(t)
            
            # 添加噪声
            noise = torch.randn_like(x_0_sb)
            sqrt_alpha_t = torch.sqrt(dilated_t).unsqueeze(-1)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - dilated_t).unsqueeze(-1)
            
            x_t = sqrt_alpha_t * x_0_sb + sqrt_one_minus_alpha_t * noise
            
            # 预测得分
            score = self.forward(x_t, dilated_t)
            
            # 计算似然贡献
            score_norm = torch.sum(score ** 2, dim=[1, 2])
            log_likelihood = log_likelihood - 0.5 * score_norm * dt
        
        return log_likelihood


class DDSMInterface:
    """
    与现有扩散模型的接口转换器
    """
    
    def __init__(self, dirichlet_model: DirichletDiffusionModel):
        """
        Args:
            dirichlet_model: Dirichlet扩散模型实例
        """
        self.dirichlet_model = dirichlet_model
    
    def convert_from_standard_diffusion(
        self,
        x: torch.Tensor,
        format_type: str = "one_hot"
    ) -> torch.Tensor:
        """
        从标准扩散模型格式转换到Dirichlet格式
        
        Args:
            x: 输入数据
            format_type: 输入格式 ("one_hot", "categorical", "embedding")
        
        Returns:
            converted: 转换后的stick-breaking表示
        """
        if format_type == "one_hot":
            # 直接转换one-hot到stick-breaking
            return self.dirichlet_model.stick_breaking.simplex_to_stick_breaking(x)
        elif format_type == "categorical":
            # 先转换为one-hot，再转换为stick-breaking
            one_hot = F.one_hot(x.long(), num_classes=4).float()
            return self.dirichlet_model.stick_breaking.simplex_to_stick_breaking(one_hot)
        elif format_type == "embedding":
            # 假设输入是连续嵌入，需要通过softmax转换为概率
            probs = F.softmax(x, dim=-1)
            return self.dirichlet_model.stick_breaking.simplex_to_stick_breaking(probs)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def convert_to_standard_diffusion(
        self,
        x: torch.Tensor,
        format_type: str = "one_hot"
    ) -> torch.Tensor:
        """
        从Dirichlet格式转换到标准扩散模型格式
        
        Args:
            x: stick-breaking表示的数据
            format_type: 输出格式
        
        Returns:
            converted: 转换后的数据
        """
        # 先转换为概率单纯形
        simplex = self.dirichlet_model.stick_breaking.stick_breaking_to_simplex(x)
        
        if format_type == "one_hot":
            return simplex
        elif format_type == "categorical":
            # 转换为类别索引
            return torch.argmax(simplex, dim=-1)
        elif format_type == "embedding":
            # 保持为概率分布
            return simplex
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def wrap_training_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        包装训练步骤，使其兼容标准训练流程
        
        Args:
            batch: 训练批次数据
        
        Returns:
            losses: 损失字典
        """
        # 确保输入是one-hot格式
        if batch.shape[-1] != 4:
            batch = F.one_hot(batch.long(), num_classes=4).float()
        
        return self.dirichlet_model.compute_loss(batch)
    
    def wrap_sampling(
        self,
        batch_size: int,
        sequence_length: int,
        **kwargs
    ) -> torch.Tensor:
        """
        包装采样过程
        
        Args:
            batch_size: 批次大小
            sequence_length: 序列长度
            **kwargs: 其他采样参数
        
        Returns:
            samples: 生成的样本
        """
        return self.dirichlet_model.sample(batch_size, sequence_length, **kwargs)