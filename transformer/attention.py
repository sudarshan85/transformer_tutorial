#!/usr/bin/env python

__all__ = ['DotProductAttention', 'MLPAttention']

import torch, math
from torch import nn
from torch.nn import functional as F

from .masking import masked_softmax

class DotProductAttention(nn.Module):
  def __init__(self, dropout, **kwargs):
    super(DotProductAttention, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, query, key, value, valid_len=None):
    d = query.shape[-1]
    scores = torch.bmm(query, key.transpose(1, 2))
    scores /= math.sqrt(d)
    scores = masked_softmax(scores, valid_len)
    attn_wts = self.dropout(scores)
    out = torch.bmm(attn_wts, value)
    return out

class MLPAttention(nn.Module):
  def __init__(self, key_size, query_size, units, dropout, **kwargs):
    super(MLPAttention, self).__init__(**kwargs)
    self.W_k = nn.Linear(key_size, units, bias=False)
    self.W_q = nn.Linear(query_size, units, bias=False)
    self.v = nn.Linear(units, 1, bias=False)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, query, key, value, valid_len):    
    query,key = self.W_q(query),self.W_k(key)
    features = query.unsqueeze(2) + key.unsqueeze(1)
    features = torch.tanh(features)
    scores = self.v(features)
    scores = scores.squeeze(-1)
    scores = masked_softmax(scores, valid_len)
    attn_wts = self.dropout(scores)    
    out = torch.bmm(attn_wts, value)
    return out          