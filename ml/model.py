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