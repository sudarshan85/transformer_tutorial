#!/usr/bin/env python

__all__ = ['DotProductAttention', 'MLPAttention', 'MultiHeadAttention']

import torch, math
from torch import nn
from torch.nn import functional as F

from .utils import *

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

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads, dropout, bias=False, **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    self.n_heads = n_heads
    self.d_model = d_model
    self.attn = DotProductAttention(dropout)
    self.linears = clone_module(nn.Linear(d_model, d_model), 4)
    
  def forward(self, query, key, value, valid_len):
    bs, seq_len, _ = query.shape

    # 1) Do all the linear projections in batch from d_model => n_heads x (d_model/n_heads)
    # 1a) Apply linear projection
    # 1b) add single axis for getting n_heads x (d_model/n_heads)
    # 1c) swap the axis to get n_heads to second position
    # 1d) reshape result to get bs * n_heads to feed all into attn layer
    query, key, value = [l(x).reshape(bs, -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2).reshape(bs * self.n_heads, seq_len, -1) for l,x in zip(self.linears, (query, key, value))]
    
    # 2) Check for valid_len for decoder masking
    if valid_len is not None:
      valid_len = torch.repeat_interleave(valid_len, repeats=self.n_heads, dim=0)

    # 3) Apply attention on all projected vectors in batch
    out = self.attn(query, key, value, valid_len)

    # 4) "Concat" using reshape and apply final linear, inverse transform of 1
    out = out.reshape(bs, self.n_heads, seq_len, -1).transpose(1,2).reshape(bs, seq_len, -1)
    out = self.linears[-1](out)
    return out        

# class MultiHeadAttention(nn.Module):
#   def __init__(self, d_model, n_heads, dropout, bias=False, **kwargs):
#     super(MultiHeadAttention, self).__init__(**kwargs)
#     self.n_heads = n_heads
#     self.d_model = d_model
#     self.attention = DotProductAttention(dropout)
#     self.W_q = nn.Linear(d_model, d_model, bias=bias)
#     self.W_k = nn.Linear(d_model, d_model, bias=bias)
#     self.W_v = nn.Linear(d_model, d_model, bias=bias)
#     self.W_o = nn.Linear(d_model, d_model, bias=bias)
    
#   def forward(self, query, key, value, valid_len):
#     query = transpose_qkv(self.W_q(query), self.n_heads)
#     key = transpose_qkv(self.W_k(key), self.n_heads)
#     value = transpose_qkv(self.W_v(value), self.n_heads)
    
#     if valid_len is not None:
#       valid_len = torch.repeat_interleave(valid_len, repeats=self.n_heads, dim=0)
    
#     out = self.attention(query, key, value, valid_len)
#     out = transpose_output(out, self.n_heads)
#     out = self.W_o(out)
#     return out    