#!/usr/bin/env python

__all__ = ['sequence_mask', 'masked_softmax', 'transpose_qkv', 'transpose_output', 'clone_module']

import torch
from copy import deepcopy
from torch import nn
from torch.nn import functional as F

def clone_module(module, n):
  return nn.ModuleList([deepcopy(module) for _ in range(n)])

def sequence_mask(x, valid_len, value=0):
  maxlen = x.shape[1]
  mask = torch.arange((maxlen), dtype=torch.float32)[None, :] >= valid_len[:, None]
  x[mask] = value
  return x

def masked_softmax(x, valid_len):
  if valid_len is None:
    return F.softmax(x, dim=-1)
  else:
    shape = x.shape
    if valid_len.dim() == 1:
      valid_len = torch.repeat_interleave(valid_len, repeats=shape[1], dim=0)
    else:
      valid_len = valid_len.reshape(-1)
  
  x = sequence_mask(x.reshape(-1, shape[-1]), valid_len, value=-1e6)
  return F.softmax(x.reshape(shape), dim=-1)

def transpose_qkv(x, n_heads):
  x = x.reshape(x.shape[0], x.shape[1], n_heads, -1)
  x = x.permute(0, 2, 1, 3)
  x = x.reshape(-1, x.shape[2], x.shape[3])
  return x

def transpose_output(x, n_heads):
  x = x.reshape(-1, n_heads, x.shape[1], x.shape[2])
  x = x.permute(0, 2, 1, 3)
  x = x.reshape(x.shape[0], x.shape[1], -1)
  return x