import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=num_layers
        )

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, num_heads, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_layer,
            num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        output = self.transformer_decoder(tgt, memory)
        logit = self.linear(output)
        return logit

class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        return decoder_output


