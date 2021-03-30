import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.multi = MultiHead(opts)
        self.ffn = FeedForward(opts)
    def forward(self, rels, his_emb, mask_attn=None):
        pre_emb, attn = self.multi(rels, his_emb, his_emb, mask_attn)
        pre_emb = self.ffn(pre_emb)
        return pre_emb, attn
    
class MultiHead(nn.Module):
    def __init__(self, opts):
        super(MultiHead, self).__init__()
        self.opts = opts
        self.Q = nn.Linear(opts.emb_dim, opts.d_k*opts.n_head)
        self.K = nn.Linear(opts.emb_dim, opts.d_k*opts.n_head)
        self.V = nn.Linear(opts.emb_dim, opts.d_v*opts.n_head)
        
        self.fc = nn.Linear(opts.n_head * opts.d_v, opts.emb_dim, bias=False)
        self.layernorm = nn.LayerNorm(opts.emb_dim, eps=1e-6)
        self.dropout = nn.Dropout(opts.dropout)
        self.rel_attn = ScaledDot(opts)
        
    def forward(self, q, k, v, mask_attn=None):
        batch_size, residual = q.size(0), q
        q_ = self.Q(q).view(batch_size, -1, self.opts.n_head, self.opts.d_k).transpose(1, 2)
        k_ = self.K(k).view(batch_size, -1, self.opts.n_head, self.opts.d_k).transpose(1, 2)
        v_ = self.V(v).view(batch_size, -1, self.opts.n_head, self.opts.d_v).transpose(1, 2)
        if mask_attn is not None:
            mask_attn = mask_attn.unsqueeze(1).repeat(1, self.opts.n_head, 1, 1)
        output, attn = self.rel_attn(q_, k_, v_, mask_attn)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.opts.n_head*self.opts.d_k)
        output = self.fc(output)
        output = residual + output
        output = self.dropout(self.layernorm(output))
        return output.squeeze(), attn

class ScaledDot(nn.Module):
    def __init__(self, opts):
        super(ScaledDot, self).__init__()
        self.opts = opts
        self.dropout = nn.Dropout(opts.dropout)
    
    def forward(self, q, k, v, mask_attn=None):
        attn = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.opts.d_k)
        if mask_attn is not None:
            attn.masked_fill_(mask_attn, -1e9)
        attn = nn.Softmax(dim=-1)(attn)
        output = torch.matmul(attn, v)
        return output, attn
    
    
class FeedForward(nn.Module):
    def __init__(self, opts):
        super(FeedForward, self).__init__()
        self.opts = opts
        self.fc1 = nn.Linear(opts.emb_dim, opts.dim_ff)
        self.fc2 = nn.Linear(opts.dim_ff, opts.emb_dim)
        self.layernorm = nn.LayerNorm(opts.emb_dim, eps=1e-6)
        self.dropout = nn.Dropout(opts.dropout)
    
    def forward(self, output):
        residual = output
        output = self.fc2(self.gelu(self.fc1(output)))
        output = residual + output
        output = self.dropout(self.layernorm(output))
        return output
    
    def gelu(self, x):
        "Implementation of the gelu activation function by Hugging Face"
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
