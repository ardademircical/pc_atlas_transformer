# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

import torch
from torch import nn
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from configuration_atlas_model_rank_based import AtlasModelRBConfig

class AtlasModelRankBased(nn.Module):
    """
    Transformer trained on ranked-value encoded prostate cancer single-cell RNA sequences.
    Self-supervision task mimics that of Geneformer.

    Dimension key:

    B: batch size
    L: sequence length
    M: memory length (length of sequence being attended to)
    D: model dimension (sometimes called d_model or embedding_dim)
    V: vocabulary size
    F: feed-forward subnetwork hidden size
    H: number of attention heads in a layer
    K: size of each attention key or value (sometimes called d_kv)

    """

    def __init__(self, vocab_size, embed_dim=256, max_seq_len=512, num_layers=6, dropout=0.1):
        super(AtlasModelRankBased, self).__init__()
        self.embeddings = AtlasEmbeddingsRB(vocab_size, embed_dim, max_seq_len, dropout)
        self.encoders = nn.ModuleList([AtlasEncoderRB() for _ in range(num_layers)])
        self.lm_head = AtlasLMHeadRB(vocab_size, embed_dim)

    def forward(self, input_ids_BL, attn_mask=None):
        embeddings_BLD = self.embeddings(input_ids_BL)
        for encoder in self.encoders:
            embeddings_BLD = encoder(embeddings_BLD, attn_mask)
        
        preds_BLV = self.lm_head(embeddings_BLD)
        return preds_BLV

class AtlasEmbeddingsRB(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, max_seq_len=512, dropout=0.1):
        super(AtlasEmbeddingsRB, self).__init__()
        self.gene_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids_BL):
        seq_length = input_ids_BL.size(1)
        position_ids_1L = torch.arange(seq_length, dtype=torch.long, device=input_ids_BL.device).unsqueeze(0)

        gene_embeddings_BLD = self.gene_embeddings(input_ids_BL)
        pos_embeddings_BLD = self.pos_embeddings(position_ids_1L)
        embeddings_BLD = gene_embeddings_BLD + pos_embeddings_BLD
        embeddings_BLD = self.layer_norm(embeddings_BLD)
        embeddings_BLD = self.dropout(embeddings_BLD)

        return embeddings_BLD
        

class AtlasEncoderRB(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, forward_expansion=4, dropout=0.1):
        super(AtlasEncoderRB, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * forward_expansion),
            nn.GELU(),
            nn.Linear(embed_dim * forward_expansion, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embeddings_BLD, attn_mask):
        x_LBD = embeddings_BLD.permute(1, 0, 2) 
        attention_output_LBD, _ = self.attention(x_LBD, x_LBD, x_LBD, attn_mask)
        attention_output_BLD = attention_output_LBD.permute(1, 0, 2) 
        x_BLD = self.layer_norm1(attention_output_BLD + embeddings_BLD)
        x_BLD = self.dropout(x_BLD)
        ff_output_BLD = self.feed_forward(x_BLD)
        x_BLD = self.layer_norm2(ff_output_BLD + x_BLD)
        output_BLD = self.dropout(x_BLD)
        return output_BLD
    

class AtlasLMHeadRB(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super(AtlasLMHeadRB, self).__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, features_BLD):
        x_BLD = self.dense(features_BLD)
        x_BLD = self.layer_norm(x_BLD)
        preds_BLV = self.decoder(x_BLD)
        return preds_BLV

