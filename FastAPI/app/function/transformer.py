import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        
        # Q, K, V を計算
        Q = self.q_linear(x)  # (batch_size, seq_length, embed_dim)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # ヘッドに分割（batch, seq_len, heads, head_dim） → (batch, heads, seq_len, head_dim) にpermute
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        # スケールド・ドット積注意
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, heads, seq_len, seq_len)
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V)  # (batch, heads, seq_len, head_dim)
        
        # ヘッドを結合 (batch, seq_len, embed_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, embed_dim)
        
        # 最終線形変換
        out = self.out_linear(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 自己注意＋残差＋正規化
        attn_out = self.self_attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # フィードフォワード＋残差＋正規化
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


if __name__ == "__main__":
    batch_size = 2
    seq_length = 10
    embed_dim = 32
    num_heads = 4
    ff_dim = 64
    
    # ランダムな入力埋め込み (例: バッチ×系列長×埋め込み次元)
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Transformer Encoder層を作成
    encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, ff_dim)

    # 順伝搬
    output = encoder_layer(x)
    print(output.shape)  # => (2, 10, 32)
