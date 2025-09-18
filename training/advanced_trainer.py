# -*- coding: utf-8 -*-
"""高级训练器模块

集成分布式训练支持、先进优化策略、梯度裁剪和混合精度训练
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import time
import logging
import json
import os
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager

class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict[str, Any],
                 device: torch.device,
                 logger: Optional[logging.Logger] = None):
        
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # 分布式训练设置
        self.is_distributed = config.get('distributed', False)
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 梯度裁剪
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # 训练配置
        self.num_epochs = config.get('num_epochs', 100)
        self.save_interval = config.get('save_interval', 10)
        self.eval_interval = config.get('eval_interval', 5)
        self.log_interval = config.get('log_interval', 100)
        
        # 早停配置
        self.early_stopping_patience = config.get('early_stopping_patience', 20)
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # 输出目录
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练统计
        self.training_stats = defaultdict(list)
        
        # 初始化优化器和调度器
        self._setup_optimizer_and_scheduler()
        
        # 分布式训练初始化
        if self.is_distributed:
            self._setup_distributed_training()
            
    def _setup_optimizer_and_scheduler(self):
        """设置优化器和学习率调度器"""
        
        # AdamW优化器配置
        optimizer_config = self.config.get('optimizer', {})
        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 1e-2)
        betas = optimizer_config.get('betas', (0.9, 0.999))
        eps = optimizer_config.get('eps', 1e-8)
        
        # 参数分组（不同参数组使用不同的权重衰减）
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps
        )
        
        # 学习率调度器配置
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', self.num_epochs)
            eta_min = scheduler_config.get('eta_min', lr * 0.01)
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_type == 'onecycle':
            max_lr = scheduler_config.get('max_lr', lr * 10)
            self.scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=max_lr, 
                epochs=self.num_epochs,
                steps_per_epoch=scheduler_config.get('steps_per_epoch', 100)
            )
        else:
            self.scheduler = None
            
    def _setup_distributed_training(self):
        """设置分布式训练"""
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.local_rank
            )
            
        torch.cuda.set_device(self.local_rank)
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
        self.logger.info(f"分布式训练初始化完成，Rank: {self.local_rank}, World Size: {self.world_size}")
        
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   criterion: nn.Module,
                   epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        epoch_stats = defaultdict(list)
        
        total_batches = len(train_loader)
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据准备
            batch = self._prepare_batch(batch)
            
            # 前向传播
            with autocast(enabled=self.use_amp):
                outputs = self.model(**batch)
                loss = criterion(outputs, batch)
                
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    
                self.optimizer.step()
                
            # 学习率调度
            if self.scheduler and self.config.get('scheduler', {}).get('type') == 'onecycle':
                self.scheduler.step()
                
            # 统计信息更新
            epoch_stats['loss'].append(loss.item())
            epoch_stats['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # 日志记录
            if (batch_idx + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx+1}/{total_batches}, "
                    f"Loss: {loss.item():.6f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                    f"Speed: {batches_per_sec:.2f} batches/sec"
                )
                
        # Epoch结束后的学习率调度
        if self.scheduler and self.config.get('scheduler', {}).get('type') != 'onecycle':
            self.scheduler.step()
            
        # 返回epoch统计信息
        return {
            'loss': np.mean(epoch_stats['loss']),
            'lr': epoch_stats['lr'][-1] if epoch_stats['lr'] else 0.0
        }
        
    def evaluate(self, 
                val_loader: DataLoader, 
                criterion: nn.Module,
                metrics: Optional[List[Callable]] = None) -> Dict[str, float]:
        """评估模型"""
        
        self.model.eval()
        eval_stats = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._prepare_batch(batch)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(**batch)
                    loss = criterion(outputs, batch)
                    
                eval_stats['loss'].append(loss.item())
                
                # 计算额外指标
                if metrics:
                    for metric_fn in metrics:
                        metric_value = metric_fn(outputs, batch)
                        metric_name = getattr(metric_fn, '__name__', 'metric')
                        eval_stats[metric_name].append(metric_value)
                        
        # 返回评估结果
        results = {}
        for key, values in eval_stats.items():
            results[key] = np.mean(values)
            
        return results
        
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              criterion: Optional[nn.Module] = None,
              metrics: Optional[List[Callable]] = None) -> Dict[str, List[float]]:
        """完整训练流程"""
        
        if criterion is None:
            criterion = nn.MSELoss()
            
        self.logger.info("开始训练...")
        self.logger.info(f"总epochs: {self.num_epochs}")
        self.logger.info(f"使用设备: {self.device}")
        self.logger.info(f"分布式训练: {self.is_distributed}")
        self.logger.info(f"混合精度训练: {self.use_amp}")
        
        for epoch in range(1, self.num_epochs + 1):
            # 分布式训练的epoch设置
            if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
                
            # 训练一个epoch
            train_stats = self.train_epoch(train_loader, criterion, epoch)
            
            # 记录训练统计
            for key, value in train_stats.items():
                self.training_stats[f'train_{key}'].append(value)
                
            # 验证评估
            if val_loader and epoch % self.eval_interval == 0:
                val_stats = self.evaluate(val_loader, criterion, metrics)
                
                # 记录验证统计
                for key, value in val_stats.items():
                    self.training_stats[f'val_{key}'].append(value)
                    
                # 早停检查
                current_metric = val_stats.get('loss', float('inf'))
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    
                    # 保存最佳模型
                    if self._is_main_process():
                        self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    
                # 早停
                if self.patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"早停触发，在epoch {epoch}停止训练")
                    break
                    
                self.logger.info(
                    f"Epoch {epoch} - Train Loss: {train_stats['loss']:.6f}, "
                    f"Val Loss: {val_stats.get('loss', 0):.6f}, "
                    f"Best: {self.best_metric:.6f}, Patience: {self.patience_counter}"
                )
                
            # 定期保存检查点
            if epoch % self.save_interval == 0 and self._is_main_process():
                self._save_checkpoint(epoch)
                
        self.logger.info("训练完成！")
        return dict(self.training_stats)
        
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """准备批次数据"""
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device, non_blocking=True)
            else:
                prepared_batch[key] = value
        return prepared_batch
        
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'training_stats': dict(self.training_stats)
        }
        
        # 保存常规检查点
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
            
        self.logger.info(f"保存检查点到: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.training_stats = defaultdict(list, checkpoint.get('training_stats', {}))
        
        self.logger.info(f"从 {checkpoint_path} 加载检查点成功")
        return checkpoint['epoch']
        
    def _is_main_process(self) -> bool:
        """判断是否为主进程"""
        return not self.is_distributed or self.local_rank == 0
        
    def get_training_stats(self) -> Dict[str, List[float]]:
        """获取训练统计信息"""
        return dict(self.training_stats)
        
    def save_training_stats(self, filename: str = 'training_stats.json'):
        """保存训练统计信息"""
        if self._is_main_process():
            stats_path = self.output_dir / filename
            with open(stats_path, 'w') as f:
                json.dump(dict(self.training_stats), f, indent=2)
            self.logger.info(f"训练统计信息保存到: {stats_path}")

@contextmanager
def distributed_training_context(local_rank: int):
    """分布式训练上下文管理器"""
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def create_advanced_trainer(model: nn.Module, 
                          config: Dict[str, Any],
                          device: torch.device,
                          logger: Optional[logging.Logger] = None) -> AdvancedTrainer:
    """创建高级训练器"""
    return AdvancedTrainer(model, config, device, logger)

if __name__ == "__main__":
    # 测试示例
    from torch.utils.data import TensorDataset, DataLoader
    
    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 1)
            
        def forward(self, input_ids, **kwargs):
            return {'predictions': self.linear(input_ids.float())}
            
    # 创建测试数据
    test_data = torch.randn(1000, 100)
    test_targets = torch.randn(1000, 1)
    dataset = TensorDataset(test_data, test_targets)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 训练配置
    config = {
        'num_epochs': 10,
        'use_amp': True,
        'max_grad_norm': 1.0,
        'optimizer': {
            'lr': 1e-3,
            'weight_decay': 1e-2
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 10
        },
        'output_dir': 'test_outputs'
    }
    
    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TestModel().to(device)
    trainer = create_advanced_trainer(model, config, device)
    
    # 自定义损失函数
    def custom_criterion(outputs, batch):
        predictions = outputs['predictions']
        targets = batch['labels' if 'labels' in batch else 1]  # 使用targets作为labels
        return nn.MSELoss()(predictions, targets)
    
    print("开始测试训练...")
    # 准备批次数据格式
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        return {
            'input_ids': torch.stack(inputs),
            'labels': torch.stack(targets)
        }
        
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # 开始训练
    stats = trainer.train(train_loader, criterion=custom_criterion)
    print(f"训练完成，最终损失: {stats['train_loss'][-1]:.6f}")
