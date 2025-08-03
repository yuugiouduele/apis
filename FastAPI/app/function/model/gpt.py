import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        
        Q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)  # (B,H,L,L)
        
        # 未来の位置をマスクする下三角マスク(triuで上三角が1なので逆に使う)
        mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)  # (L,L)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B,H,L,D)
        
        context = context.transpose(1,2).contiguous().view(batch_size, seq_length, embed_dim)
        out = self.out_linear(context)
        return out

# 使い方例
batch_size = 2
seq_length = 5
embed_dim = 32
num_heads = 4

x = torch.rand(batch_size, seq_length, embed_dim)

model = GPTSelfAttention(embed_dim, num_heads)
out = model(x)
print(out.shape)  # (2, 5, 32)
