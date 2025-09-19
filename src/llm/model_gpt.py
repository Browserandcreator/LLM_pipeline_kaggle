import math, torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, x): return F.gelu(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B,T,self.n_head,self.head_dim).transpose(1,2)  # (B,nh,T,hd)
        k = k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_head,self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * self.scale  # (B,nh,T,T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B,nh,T,hd)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, int(mlp_ratio*n_embd)),
            GELU(),
            nn.Linear(int(mlp_ratio*n_embd), n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_head=8, n_embd=512, block_size=256, dropout=0.1, tie_weights=True):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1,T)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
