"""文件IO工具"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """保存JSON文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Saved JSON file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {filepath}: {e}")
        raise


def load_json(filepath: Union[str, Path]) -> Any:
    """加载JSON文件"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON file: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {filepath}: {e}")
        raise


def save_fasta(sequences: Dict[str, str], filepath: Union[str, Path]) -> None:
    """保存FASTA文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for header, sequence in sequences.items():
                f.write(f">{header}\n")
                # 每行80个字符
                for i in range(0, len(sequence), 80):
                    f.write(f"{sequence[i:i+80]}\n")
        logger.debug(f"Saved FASTA file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save FASTA file {filepath}: {e}")
        raise


def load_fasta(filepath: Union[str, Path]) -> Dict[str, str]:
    """加载FASTA文件"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"FASTA file not found: {filepath}")
    
    sequences = {}
    current_header = None
    current_sequence = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('>'):
                    # 保存上一个序列
                    if current_header is not None:
                        sequences[current_header] = ''.join(current_sequence)
                    
                    # 开始新序列
                    current_header = line[1:]  # 去掉'>'
                    current_sequence = []
                else:
                    current_sequence.append(line)
            
            # 保存最后一个序列
            if current_header is not None:
                sequences[current_header] = ''.join(current_sequence)
        
        logger.debug(f"Loaded FASTA file: {filepath} ({len(sequences)} sequences)")
        return sequences
    except Exception as e:
        logger.error(f"Failed to load FASTA file {filepath}: {e}")
        raise


def save_csv(data: Union[List[Dict], pd.DataFrame], filepath: Union[str, Path], 
             index: bool = False) -> None:
    """保存CSV文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        df.to_csv(filepath, index=index, encoding='utf-8')
        logger.debug(f"Saved CSV file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save CSV file {filepath}: {e}")
        raise


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """加载CSV文件"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        logger.debug(f"Loaded CSV file: {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file {filepath}: {e}")
        raise
