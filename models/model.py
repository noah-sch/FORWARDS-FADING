# Lib import 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from dataclasses import dataclass
from typing import Optionnal 
from tdqm import tdqm

# Open config 
import yaml
with open("config.yml", "r") as config_file: 
    config = yaml.safe_load(config_file)

### CLASS 
@dataclass 
class ModelConfig: 
    vocab_size: int = config["model"]["config"]["vocab_size"]
    d_model: int = config["model"]["config"]["dim_embeddings"]
    n_heads: int = config["model"]["config"]["nb_heads"]
    n_layers: int = config["model"]["config"]["nb_layers"]
    d_ff: int = config["model"]["config"]["dim_ff"]
    max_seq_len: int = config["model"]["config"]["max_seq_len"]
    dropout: float = config["model"]["config"]["dropout"]


### MULTIHEAD ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0 

        # Generals
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        # Q, K, V
        self.W_q = nn.Linerar(config.d_model, config.d_model)
        self.W_k = nn.Linerar(config.d_model, config.d_model)
        self.W_v = nn.Linerar(config.d_model, config.d_model)

        # Output proj
        self.W_o = nn.Linerar(config.d_model, config.d_model)

        # Dropout 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None): 
        batch_size, seq_len, d_model = x.shape

        # Proj + multihead reshape
        Q = self.W_q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1) / math.sqrt(self.d_k))

        # Causal attention mask
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Dropout
        attention_weights = self.dropout(attention_weights)

        # Attention to values
        output = torch.matmul(attention_weights, V)

        # Reshape + final proj
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(output)

        # Return 
        return output


### FEEDFORWARD NETWORK

class FeedForward(nn.Module): 
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x): 
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x 


### TRANSFORMER BLOCK
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None): 
        # Attention 
        attn_o = self.attention(self.norm1(x), mask)
        x += self.dropout(attn_o)

        # Feedforward 
        ff_o = self.feed_forward(self.norm2(x))
        x += self.dropout(ff_o)

        # Return 
        return x 


### COMPLETE TRANSFORMER LLM
class TransformerLLM(nn.Module): 
    def __init__(self, config: ModelConfig): 
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Layers 
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)

        ])

        # Final norm + lm head
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight sharing 
        self.token_embedding.weight = self.lm_head.weight

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Weights init
        self.apply(self._init_weights)
    
    def _init_weights_(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape

        # Causal mask for attention
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(
            1, 
            1, 
            seq_len,
            seq_len
        )

        # Embeddings
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqeeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # inLayers
        for layer in self.layers: 
            x = layer(x, mask)
        
        # Final Norm
        x = self.norm(x)

        # Logit calculous
        logits = self.lm_head(x)

        # Loss calculous (if targets)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        
        # Return
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        #TOFILL
        Docstring for generate
        
        :param self: Description
        :param idx: Description
        :param max_new_tokens: Description
        :param temperature: Description
        :param top_k: Description
        """
        for _ in range(max_new_tokens): 

            # CTX trunc
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx [:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Last token logits 
            logits = logits[:, -1, :] / temperature

            # Top-k sampling 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax + echantillonnage
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Push to CTX
            idx = torch.cat((idx, idx_next), dim=1)
        
        # Return 
        return idx
    