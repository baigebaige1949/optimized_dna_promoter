"""改进的预测模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from .feature_extractor import FeatureExtractor
from ..config.model_config import PredictorModelConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DNATokenizer:
    """
DNA序列的简单tokenizer
"""
    
    def __init__(self, max_length: int = 512):
        self.nucleotides = ['A', 'T', 'G', 'C']
        self.token_to_id = {
            '<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3,
            'A': 4, 'T': 5, 'G': 6, 'C': 7
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.max_length = max_length
        
    def encode(self, sequence: str, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """编码DNA序列"""
        sequence_upper = sequence.upper()
        
        # 转换为令牌ID
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.token_to_id['<CLS>'])
        
        for nucleotide in sequence_upper:
            if nucleotide in self.nucleotides:
                token_ids.append(self.token_to_id[nucleotide])
            else:
                token_ids.append(self.token_to_id['<UNK>'])
        
        if add_special_tokens:
            token_ids.append(self.token_to_id['<SEP>'])
        
        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            attention_mask = [1] * len(token_ids)
            # 填充
            while len(token_ids) < self.max_length:
                token_ids.append(self.token_to_id['<PAD>'])
                attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def batch_encode(self, sequences: List[str], add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        """批量编码"""
        batch_encodings = [self.encode(seq, add_special_tokens) for seq in sequences]
        
        return {
            'input_ids': torch.stack([enc['input_ids'] for enc in batch_encodings]),
            'attention_mask': torch.stack([enc['attention_mask'] for enc in batch_encodings])
        }


class DNAEmbedding(nn.Module):
    """
DNA序列嵌入层
"""
    
    def __init__(self, vocab_size: int = 8, hidden_dim: int = 768, max_position: int = 512):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TransformerEncoder(nn.Module):
    """简化的Transformer编码器"""
    
    def __init__(self, hidden_dim: int = 768, num_layers: int = 6, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 转换attention_mask格式
        if attention_mask is not None:
            # Transformer期期望的是用True表示需要mask的位置
            mask = ~attention_mask.bool()
        else:
            mask = None
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=mask)
        
        return hidden_states


class PromoterPredictor(nn.Module):
    """启动子强度预测模型"""
    
    def __init__(self, config: PredictorModelConfig):
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(config)
        
        # 初始化模型组件
        if config.use_pretrained:
            try:
                # 尝试加载预训练模型
                self.pretrained_model = AutoModel.from_pretrained(config.pretrained_model_name)
                self.tokenizer_type = 'pretrained'
                logger.info(f"Loaded pretrained model: {config.pretrained_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load pretrained model: {e}. Using simple transformer.")
                self._init_simple_transformer()
        else:
            self._init_simple_transformer()
        
        # 冻结预训练模型参数
        if config.use_pretrained and config.freeze_pretrained and hasattr(self, 'pretrained_model'):
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        
        # 分类头
        self._init_classifier()
        
        logger.info(f"Initialized predictor model with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _init_simple_transformer(self):
        """初始化简单的transformer模型"""
        self.tokenizer_type = 'simple'
        self.dna_tokenizer = DNATokenizer()
        self.dna_embedding = DNAEmbedding()
        self.transformer_encoder = TransformerEncoder()
        logger.info("Initialized simple transformer encoder")
    
    def _init_classifier(self):
        """初始化分类头"""
        # 计算输入维度
        if hasattr(self, 'pretrained_model'):
            transformer_dim = self.pretrained_model.config.hidden_size
        else:
            transformer_dim = 768  # 默认维度
        
        feature_dim = self.feature_extractor.output_dim
        total_input_dim = transformer_dim + feature_dim
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(total_input_dim, self.config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden_dim, self.config.classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden_dim // 2, self.config.num_classes)
        )
    
    def encode_sequences(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码DNA序列"""
        if self.tokenizer_type == 'pretrained' and hasattr(self, 'pretrained_model'):
            # 使用预训练模型的tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name)
                encoded = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
            except:
                # 如果失败，使用简单tokenizer
                encoded = self.dna_tokenizer.batch_encode(sequences)
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
        else:
            # 使用简单tokenizer
            encoded = self.dna_tokenizer.batch_encode(sequences)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        
        return input_ids, attention_mask
    
    def get_sequence_embeddings(self, sequences: List[str]) -> torch.Tensor:
        """获取序列嵌入"""
        input_ids, attention_mask = self.encode_sequences(sequences)
        input_ids = input_ids.to(next(self.parameters()).device)
        attention_mask = attention_mask.to(next(self.parameters()).device)
        
        if hasattr(self, 'pretrained_model'):
            # 使用预训练模型
            with torch.no_grad() if self.config.freeze_pretrained else torch.enable_grad():
                outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
                # 取[CLS]令牌的表示或平均pooling
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                    # 使用[CLS]令牌（第一个令牌）或平均pooling
                    sequence_embeddings = hidden_states[:, 0, :]  # [CLS] token
                else:
                    sequence_embeddings = outputs[0][:, 0, :]  # fallback
        else:
            # 使用简单transformer
            embeddings = self.dna_embedding(input_ids)
            hidden_states = self.transformer_encoder(embeddings, attention_mask)
            # 平均pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            sequence_embeddings = sum_embeddings / sum_mask
        
        return sequence_embeddings
    
    def forward(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 获取序列嵌入
        sequence_embeddings = self.get_sequence_embeddings(sequences)
        
        # 获取手工特征
        manual_features = self.feature_extractor(sequences)
        manual_features = manual_features.to(sequence_embeddings.device)
        
        # 融合特征
        combined_features = torch.cat([sequence_embeddings, manual_features], dim=1)
        
        # 预测
        logits = self.classifier(combined_features)
        
        # 对于回归任务，使用sigmoid激活
        if self.config.num_classes == 1:
            predictions = torch.sigmoid(logits.squeeze(-1))  # 输出0-1之间的值
        else:
            predictions = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'predictions': predictions,
            'sequence_embeddings': sequence_embeddings,
            'manual_features': manual_features
        }
    
    def predict_strength(self, sequences: List[str]) -> List[float]:
        """预测启动子强度"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sequences)
            predictions = outputs['predictions']
            
            if isinstance(predictions, torch.Tensor):
                return predictions.cpu().numpy().tolist()
            else:
                return predictions.tolist()
    
    def get_feature_importance(self, sequences: List[str]) -> Dict[str, float]:
        """获取特征重要性（简单实现）"""
        self.eval()
        
        # 这里可以实现更复杂的特征重要性分析
        # 目前返回一个简单的实现
        
        importance = {}
        
        # 模拟特征重要性分数
        for k in self.config.kmer_sizes:
            importance[f'kmer_{k}'] = 0.8 - 0.1 * (k - 3)  # k越小重要性越高
        
        if self.config.use_biological_features:
            importance['biological'] = 0.9
        
        importance['sequence_model'] = 0.95
        
        return importance
