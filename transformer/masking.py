#!/usr/bin/env python

__all__ = ['sequence_mask', 'masked_softmax']

import torch
from torch.nn import functional as F

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