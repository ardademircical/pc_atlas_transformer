import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size=768, heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(embed_size, heads * self.head_dim, bias=False)
        self.keys = nn.Linear(embed_size, heads * self.head_dim, bias=False)
        self.queries = nn.Linear(embed_size, heads * self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def split_into_heads(self, x, batch_size):
        # Split the last dimension into (heads x head_dim).
        # Then reshape to (batch_size x heads x sequence_length x head_dim)
        x = x.view(batch_size, -1, self.heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, values, keys, query, mask):
        batch_size = query.shape[0]

        values = self.split_into_heads(self.values(values), batch_size)
        keys = self.split_into_heads(self.keys(keys), batch_size)
        queries = self.split_into_heads(self.queries(query), batch_size)

        # Einsum does matrix multiplication for query and keys for each training example
        # and each head (bmm is batch matrix multiplication)
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch_size, -1, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size=768, heads=8, dropout=0.1, forward_expansion=4):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class AtlasTransformerMultiClassifier(nn.Module):
    def __init__(self, original_embed_size, embed_size=768, num_layers=8, heads=8, device=None, forward_expansion=4, dropout=0.1, num_classes=1):
        super(AtlasTransformerMultiClassifier, self).__init__()
        self.input_projection = nn.Linear(original_embed_size, embed_size)
        self.encoder = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.device = device
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_size, num_classes)

    def forward(self, x, mask=None):
        # Project input embeddings to size 768
        x = self.input_projection(x)
        for layer in self.encoder:
            x = layer(x, x, x, mask)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(2)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        # out = torch.softmax(out, dim=-1)  # Change to softmax for multi-class
        return out
