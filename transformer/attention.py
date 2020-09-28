#!/usr/bin/env python

__all__ = ['DotProductAttention',]

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