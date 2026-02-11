import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size):
        super().__init__()
        
        self.block_size = block_size
        self.embed_dim = 384
        
        self.token_emb = nn.Embedding(vocab_size, self.embed_dim)
        self.pos_emb = nn.Embedding(block_size, self.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=6,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        
        self.ln = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, vocab_size)
        
    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)
        
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        
        x = self.ln(x)
        logits = self.head(x)
        
        return logits