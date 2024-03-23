# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

import torch
from torch import nn
import numpy as np
from typing import Optional

class ModelArgs:
    def __init__(self, embed_dim: int = 512, num_layers: int = 6,
                num_heads: int = 8, vocab_size: int = 13748, 
                norm_eps: float = 1e-6, max_seq_len: int = 4096, 
                dropout: float = 0.1, forward_expansion: int = 4
        ):
        """
        Args:
            embed_dim (int): The dimension of the embedding.
            num_layers (int): The number of layers in the model.
            num_heads (int): The number of attention heads in the model.
            vocab_size (int): The size of the vocabulary.
            norm_eps (float): The epsilon value for normalization.
            max_seq_len (int): The maximum sequence length.
            dropout (float): The dropout rate.
            forward_expansion (int): The forward expansion factor.
        """
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.forward_expansion = forward_expansion

class AtlasEmbeddingsRB(nn.Module):
    def __init__(self, args: ModelArgs):
        super(AtlasEmbeddingsRB, self).__init__()

        self.vocab_size = args.vocab_size
        self.embed_dim = args.embed_dim 
        self.max_seq_len = args.max_seq_len
        self.gene_embeddings = nn.Embedding(args.vocab_size, args.embed_dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.embed_dim)
#         self.norm = nn.LayerNorm(args.embed_dim)
        self.norm = RMSNorm(args.embed_dim, args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_ids_BL):
        seq_length = input_ids_BL.size(1)
        position_ids_1L = torch.arange(seq_length, dtype=torch.long, device=input_ids_BL.device).unsqueeze(0)

        gene_embeddings_BLD = self.gene_embeddings(input_ids_BL)
        pos_embeddings_BLD = self.pos_embeddings(position_ids_1L)
        embeddings_BLD = gene_embeddings_BLD + pos_embeddings_BLD
        embeddings_BLD = self.norm(embeddings_BLD)
        embeddings_BLD = self.dropout(embeddings_BLD)

        return embeddings_BLD
        

class AtlasEncoderRB(nn.Module):
    def __init__(self, args: ModelArgs):
        super(AtlasEncoderRB, self).__init__()
        self.embed_dim = args.embed_dim 
        self.num_heads = args.num_heads
        self.forward_expansion = args.forward_expansion
        self.attention = nn.MultiheadAttention(args.embed_dim, args.num_heads, args.dropout, kdim=args.embed_dim, vdim=args.embed_dim)

#         self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm1 = RMSNorm(args.embed_dim, args.norm_eps)
        
        
        self.feed_forward = nn.Sequential(
            nn.Linear(args.embed_dim, args.embed_dim * args.forward_expansion),
            nn.SiLU(),
            nn.Linear(args.embed_dim * args.forward_expansion, args.embed_dim)
        )
#         self.norm2 = nn.LayerNorm(args.embed_dim)
        self.norm2 = RMSNorm(args.embed_dim, args.norm_eps)
        
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, embeddings_BLD, key_padding_mask):
        x_LBD = embeddings_BLD.permute(1, 0, 2) 
        attention_output_LBD, _ = self.attention(x_LBD, x_LBD, x_LBD, key_padding_mask=key_padding_mask)
        attention_output_BLD = attention_output_LBD.permute(1, 0, 2) 
        x_BLD = self.norm1(attention_output_BLD + embeddings_BLD)
        x_BLD = self.dropout(x_BLD)
        ff_output_BLD = self.feed_forward(x_BLD)
        x_BLD = self.norm2(ff_output_BLD + x_BLD)
        output_BLD = self.dropout(x_BLD)
        return output_BLD
    

class AtlasLMHeadRB(nn.Module):
    def __init__(self, args: ModelArgs):
        super(AtlasLMHeadRB, self).__init__()
        self.dense = nn.Linear(args.embed_dim, args.embed_dim)
#         self.norm = nn.LayerNorm(args.embed_dim)
        self.norm = RMSNorm(args.embed_dim, args.norm_eps)
        self.decoder = nn.Linear(args.embed_dim, args.vocab_size)
    
    def forward(self, features_BLD):
        x_BLD = self.dense(features_BLD)
        x_BLD = self.norm(x_BLD)
        preds_BLV = self.decoder(x_BLD)
        return preds_BLV


class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int = 512, norm_eps: float = 1e-5):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class AtlasModelRankBased(nn.Module):
    """
    Transformer trained on ranked-value encoded prostate cancer single-cell RNA sequences.
    Self-supervision task mimics that of Geneformer.

    Dimension key:

    B: batch size
    L: sequence length
    D: embedding dimension 
    V: vocabulary size
    F: feed-forward subnetwork hidden size
    H: number of attention heads in a layer

    """

    def __init__(self, params: ModelArgs):
        super(AtlasModelRankBased, self).__init__()
        self.params = params
        self.embeddings = AtlasEmbeddingsRB(params)
        self.encoders = nn.ModuleList([AtlasEncoderRB(params) for _ in range(params.num_layers)])
        self.lm_head = AtlasLMHeadRB(params)

    def forward(self, input_ids_BL, key_padding_mask, attn_mask=None):
        embeddings_BLD = self.embeddings(input_ids_BL)
        for encoder in self.encoders:
            embeddings_BLD = encoder(embeddings_BLD,key_padding_mask)
        
        preds_BLV = self.lm_head(embeddings_BLD)
        return preds_BLV
