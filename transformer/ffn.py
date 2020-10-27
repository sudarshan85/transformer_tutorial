#!/usr/bin/env python

__all__ = ['PositionWiseFFN', 'AddNorm']

import torch
from torch import nn

class PositionWiseFFN(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
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