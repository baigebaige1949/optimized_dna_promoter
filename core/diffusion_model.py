"""改进的扩散模型实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..config.model_config import DiffusionModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NoiseScheduler:
    """噪声调度器，管理扩散过程中的噪声谈度表"""
    
    def __init__(self, config: DiffusionModelConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        
        # 生成beta谈度表
        if config.beta_schedule == "linear":
            self.betas = torch.linspace(config.beta_start, config.beta_end, config.num_timesteps)
        elif config.beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")
        
        # 预计算常用系数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 逆向过程参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 方差参数
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        logger.info(f"Initialized noise scheduler with {config.num_timesteps} timesteps, {config.beta_schedule} schedule")
    
    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """余弦噪声谈度表"""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """在原始数据上添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
        return noisy_x, noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
    
    def to(self, device: torch.device) -> 'NoiseScheduler':
        """移动到指定设备"""
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'posterior_variance']:
            setattr(self, attr_name, getattr(self, attr_name).to(device))
        return self


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """改进的残差块，U-Net的基本组件"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 dropout: float = 0.1, use_attention: bool = False):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
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
        
        # 可选的自注意力机制
        self.attention = (
            SelfAttention1D(out_channels) 
            if use_attention 
            else None
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        
        # 添加时间嵌入
        time_emb = self.time_mlp(time_emb)[:, :, None]
        h = h + time_emb
        
        h = self.block2(h)
        
        # 残差连接
        h = h + self.shortcut(x)
        
        # 自注意力
        if self.attention is not None:
            h = self.attention(h)
        
        return h


class SelfAttention1D(nn.Module):
    """一维自注意力机制"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h).reshape(b, 3, self.num_heads, self.head_dim, l)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # 计算注意力分数
        attn = torch.einsum('bhdi,bhdj->bhij', q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(b, c, l)
        out = self.proj_out(out)
        
        return x + out


class UNet1D(nn.Module):
    """一维U-Net模型"""
    
    def __init__(self, config: DiffusionModelConfig, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 时间嵌入
        time_emb_dim = config.hidden_dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 条件嵌入（可选）
        if config.use_condition:
            self.condition_embedding = nn.Linear(config.condition_dim, time_emb_dim)
        
        # 编码器
        self.input_conv = nn.Conv1d(in_channels, config.unet_channels[0], kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        channels = config.unet_channels
        for i in range(len(channels) - 1):
            self.down_blocks.append(ResidualBlock(
                channels[i], channels[i+1], time_emb_dim,
                config.dropout, use_attention=(channels[i+1] >= 128)
            ))
        
        # 中间层
        mid_channels = channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, time_emb_dim, config.dropout, True)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, time_emb_dim, config.dropout, True)
        
        # 解码器
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(ResidualBlock(
                channels[i] + channels[i-1], channels[i-1], time_emb_dim,
                config.dropout, use_attention=(channels[i-1] >= 128)
            ))
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 时间嵌入
        time_emb = self.time_embedding(timesteps)
        
        # 条件嵌入
        if condition is not None and hasattr(self, 'condition_embedding'):
            condition_emb = self.condition_embedding(condition)
            time_emb = time_emb + condition_emb
        
        # 输入层
        h = self.input_conv(x)
        
        # 编码器
        skip_connections = [h]
        for block in self.down_blocks:
            h = block(h, time_emb)
            skip_connections.append(h)
            h = F.avg_pool1d(h, kernel_size=2)
        
        # 中间层
        h = self.mid_block1(h, time_emb)
        h = self.mid_block2(h, time_emb)
        
        # 解码器
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            h = torch.cat([h, skip_connections.pop()], dim=1)
            h = block(h, time_emb)
        
        # 输出层
        return self.output_conv(h)


class DiffusionModel(nn.Module):
    """主扩散模型"""
    
    def __init__(self, config: DiffusionModelConfig, vocab_size: int = 4):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # 噪声调度器
        self.noise_scheduler = NoiseScheduler(config)
        
        # U-Net模型
        self.unet = UNet1D(config, vocab_size, vocab_size)
        
        logger.info(f"Initialized diffusion model with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """训练时的前向传播"""
        batch_size = x.size(0)
        device = x.device
        
        # 采样时间步和噪壳
        timesteps = self.noise_scheduler.sample_timesteps(batch_size, device)
        noise = torch.randn_like(x)
        
        # 添加噪壱
        noisy_x, _ = self.noise_scheduler.add_noise(x, timesteps, noise)
        
        # 预测噪壳
        predicted_noise = self.unet(noisy_x, timesteps, condition)
        
        return {
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'timesteps': timesteps
        }
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], condition: Optional[torch.Tensor] = None,
               num_inference_steps: int = 50, device: torch.device = None) -> torch.Tensor:
        """生成样本"""
        if device is None:
            device = next(self.parameters()).device
        
        # 初始化为随机噪壳
        x = torch.randn(shape, device=device)
        
        # 计算推理步骤
        timesteps = torch.linspace(self.config.num_timesteps - 1, 0, num_inference_steps, device=device).long()
        
        for t in timesteps:
            # 预测噪壳
            t_batch = t.expand(shape[0])
            predicted_noise = self.unet(x, t_batch, condition)
            
            # 去噪壳
            alpha = self.noise_scheduler.alphas[t]
            alpha_cumprod = self.noise_scheduler.alphas_cumprod[t]
            beta = self.noise_scheduler.betas[t]
            
            # DDPM采样公式
            x = (x - beta * predicted_noise / torch.sqrt(1 - alpha_cumprod)) / torch.sqrt(alpha)
            
            # 添加噪壳（除了最后一步）
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta) * noise
        
        return x
    
    def to(self, device: torch.device) -> 'DiffusionModel':
        """移动到指定设备"""
        super().to(device)
        self.noise_scheduler.to(device)
        return self
