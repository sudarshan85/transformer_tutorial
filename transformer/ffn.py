#!/usr/bin/env python

__all__ = ['PositionWiseFFN', 'AddNorm', 'PositionalEncoding']

import math
import torch
from torch import nn

class PositionWiseFFN(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0, **kwargs):
    super(PositionWiseFFN, self).__init__(**kwargs)
    self.model = nn.Sequential(
      nn.Linear(d_model, d_ff),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(d_ff, d_model)
    )

  def forward(self, x):
    return self.model(x)

class AddNorm(nn.Module):
  def __init__(self, features, dropout=0., **kwargs):
    super(AddNorm, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)
    self.ln = nn.LayerNorm(features)
    
  def forward(self, x, y):
    return self.ln(self.dropout(y) + x)    

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0., max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(dropout)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.pe = pe.unsqueeze(0)
    
  def forward(self, x):
    # TODO: potential problem with requires_grad
    x = x + self.pe[:, :x.shape[1], :]
    return self.dropout(x)