import torch
import torch.nn as nn
from setting import *
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        # d_model表示embedding的维度，max_len表示句子的最大长度
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # 不用梯度下降，但是需要保存为参数

    def forward(self, x):
        # x: [batch, sequence, embedding]
        # print(x.shape)
        x = x + self.pe[0, :x.size(1), :]  # 长度不够就截断。注意batch部分触发了广播机制
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(
            input_dim, hidden_dim, padding_idx=padding_token)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, tgt, mask_in, mask_out):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        seq_len = tgt.size(1)
        mask = torch.stack([torch.tensor([False]*i+[True]*(seq_len-i))
                           for i in range(1, seq_len+1)])
        mask = mask.to(device)
        output = self.transformer(src, tgt, tgt_mask=mask,
                                  src_key_padding_mask=mask_in, tgt_key_padding_mask=mask_out,
                                  tgt_is_causal=True)
        logit = self.linear(output)
        return logit
