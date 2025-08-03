import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BertSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_length, embed_dim = x.size()
        
        Q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)  # (B, H, L, D)
        K = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)
        
        if attention_mask is not None:
            # attention_mask: (B, 1, 1, L) or (B, 1, L, L), 0=masked, 1=keep
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B, H, L, D)
        
        context = context.transpose(1,2).contiguous().view(batch_size, seq_length, embed_dim)
        out = self.out_linear(context)
        return out

# 使い方例
batch_size = 2
seq_length = 5
embed_dim = 32
num_heads = 4

# 入力埋め込みの例
x = torch.rand(batch_size, seq_length, embed_dim)

# パディングマスク例（1=有効、0=パディング）
pad_mask = torch.tensor([
    [1,1,1,0,0],
    [1,1,1,1,0],
], dtype=torch.uint8)

# BERT系attention_maskは (batch, 1, 1, seq_len)に変形
attention_mask = pad_mask.unsqueeze(1).unsqueeze(2)

model = BertSelfAttention(embed_dim, num_heads)
out = model(x, attention_mask)

print(out.shape)  # (2, 5, 32)
