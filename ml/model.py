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
    dim_embeddings: int = config["model"]["config"]["dim_embeddings"]
    nb_heads: int = config["model"]["config"]["nb_heads"]
    nb_layers: int = config["model"]["config"]["nb_layers"]
    dim_ff: int = config["model"]["config"]["dim_ff"]
    max_seq_len: int = config["model"]["config"]["max_seq_len"]
    dropout: float = config["model"]["config"]["dropout"]


### MULTIHEAD ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.dim_embeddings % config.nb_heads == 0 

        self.dim_embeddings = config.dim_embeddings
        self.nb_heads = config.nb_heads
        self.dim_k = config.dim_embeddings // config.nb_heads

        