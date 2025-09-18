"""
Dirichlet扩散模型训练器
集成了高效训练、评估和采样功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm

from ..core.dirichlet_diffusion import DirichletDiffusionModel, DDSMInterface
from ..config.dirichlet_config import DirichletDiffusionConfig
from ..utils.logger import Logger
from ..utils.device_manager import DeviceManager

class DirichletTrainer:
    """Dirichlet扩散模型训练器"""
    
    def __init__(self, config: DirichletDiffusionConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger or Logger("DirichletTrainer")
        self.device_manager = DeviceManager()
        
        # 设置设备
        self.device = self.device_manager.get_device(config.device)
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = DirichletDiffusionModel(
            sequence_length=config.sequence_length,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            alpha=config.alpha,
            beta=config.beta,
            dilation_factor=config.dilation_factor
        ).to(self.device)
        
        # 创建接口
        self.interface = DDSMInterface(self.model)
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据预处理
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # 假设批次数据是元组形式
            
            batch = batch.to(self.device)
            
            # 确保数据格式正确
            if batch.dim() == 2:  # (batch, seq_len)
                batch = torch.nn.functional.one_hot(batch.long(), num_classes=4).float()
            
            # 前向传播
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    losses = self.model.compute_loss(batch)
                    loss = losses['total_loss']
                
                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model.compute_loss(batch)
                loss = losses['total_loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return {'train_loss': np.mean(epoch_losses)}
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        eval_losses = []
        
        for batch in tqdm(dataloader, desc="评估中"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(self.device)
            
            if batch.dim() == 2:
                batch = torch.nn.functional.one_hot(batch.long(), num_classes=4).float()
            
            losses = self.model.compute_loss(batch)
            eval_losses.append(losses['total_loss'].item())
        
        return {'eval_loss': np.mean(eval_losses)}
    
    @torch.no_grad() 
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """生成样本序列"""
        self.model.eval()
        
        samples = self.model.sample(
            batch_size=num_samples,
            sequence_length=self.config.sequence_length,
            num_steps=self.config.num_sampling_steps,
            temperature=self.config.temperature
        )
        
        return samples
    
    def analyze_generation_quality(self, samples: torch.Tensor) -> Dict[str, float]:
        """分析生成质量"""
        # 转换为类别表示
        categorical = torch.argmax(samples, dim=-1)
        
        # 碱基频率分析
        base_counts = torch.bincount(categorical.flatten(), minlength=4)
        base_freqs = base_counts.float() / base_counts.sum()
        
        # GC含量
        gc_content = (base_freqs[2] + base_freqs[3]).item()
        
        # 序列多样性
        unique_seqs = len(set(tuple(seq.tolist()) for seq in categorical))
        diversity = unique_seqs / len(categorical)
        
        return {
            'gc_content': gc_content,
            'diversity': diversity,
            'base_freq_A': base_freqs[0].item(),
            'base_freq_T': base_freqs[1].item(),
            'base_freq_G': base_freqs[2].item(),
            'base_freq_C': base_freqs[3].item()
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """完整训练流程"""
        self.logger.info("开始训练Dirichlet扩散模型")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(train_dataloader)
            self.scheduler.step()
            
            # 记录训练指标
            self.logger.log_metrics(train_metrics, step=epoch)
            
            # 验证
            if val_dataloader is not None and (epoch + 1) % self.config.eval_interval == 0:
                eval_metrics = self.evaluate(val_dataloader)
                self.logger.log_metrics(eval_metrics, step=epoch)
                
                # 保存最佳模型
                if eval_metrics['eval_loss'] < self.best_loss:
                    self.best_loss = eval_metrics['eval_loss']
                    self.save_checkpoint('best_model.pt')
                    self.logger.info(f"保存最佳模型，损失: {self.best_loss:.4f}")
            
            # 生成样本评估
            if (epoch + 1) % self.config.eval_interval == 0:
                samples = self.generate_samples(16)
                quality_metrics = self.analyze_generation_quality(samples)
                self.logger.log_metrics(quality_metrics, step=epoch)
                
                self.logger.info(f"Epoch {epoch+1}: GC含量={quality_metrics['gc_content']:.3f}, "
                               f"多样性={quality_metrics['diversity']:.3f}")
            
            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        self.logger.info("训练完成！")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filename)
        self.logger.info(f"保存检查点: {filename}")
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"加载检查点: {filename}, Epoch: {self.current_epoch}")
