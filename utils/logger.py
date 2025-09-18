"""日志系统"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import os


def setup_logging(log_level: str = "INFO", log_dir: str = "logs", 
                 log_to_file: bool = True, log_to_console: bool = True) -> None:
    """设置日志系统"""
    
    # 创建日志目录
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置日志级别
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_to_file:
        # 主日志文件
        log_filename = f"dna_promoter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = Path(log_dir) / log_filename
        
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 错误日志文件
        error_filename = f"dna_promoter_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        error_filepath = Path(log_dir) / error_filename
        
        error_handler = logging.FileHandler(error_filepath, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
    
    # 记录初始化信息
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_to_file}, Console: {log_to_console}")
    if log_to_file:
        logger.info(f"Log files: {log_filepath}, {error_filepath}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # ANSI颜色码
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = (
                self.COLORS[record.levelname] + 
                record.levelname + 
                self.COLORS['RESET']
            )
        return super().format(record)


def setup_colored_logging(log_level: str = "INFO") -> None:
    """设置带颜色的控制台日志"""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 创建带颜色的格式化器
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 设置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 添加带颜色的控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)


class PerformanceLogger:
    """性能日志器"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.start_time = None
    
    def start_timer(self, operation: str) -> None:
        """开始计时"""
        import time
        self.start_time = time.time()
        self.logger.info(f"Starting {operation}...")
    
    def end_timer(self, operation: str) -> float:
        """结束计时"""
        import time
        if self.start_time is None:
            self.logger.warning(f"Timer not started for {operation}")
            return 0.0
        
        elapsed = time.time() - self.start_time
        self.logger.info(f"Completed {operation} in {elapsed:.2f} seconds")
        self.start_time = None
        return elapsed
    
    def log_memory_usage(self, operation: str) -> None:
        """记录内存使用情况"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.logger.info(f"{operation} - Memory usage: {memory_mb:.2f} MB")
        except ImportError:
            self.logger.debug("psutil not available, skipping memory logging")
        except Exception as e:
            self.logger.warning(f"Failed to log memory usage: {e}")
