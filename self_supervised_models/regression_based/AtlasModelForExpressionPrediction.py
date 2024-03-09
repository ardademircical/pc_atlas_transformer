# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

import torch
from torch import nn
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class AtlasEncoderEP(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, forward_expansion=4, dropout=0.1):
        super(AtlasEncoderEP, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * forward_expansion),
            nn.GELU(),
            nn.Linear(embed_dim * forward_expansion, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, weighted_input_BLD, attn_mask=None):
        weighted_input_LBD = weighted_input_BLD.permute(1, 0, 2)
        attention_output_LBD, _ = self.attention(weighted_input_LBD, weighted_input_LBD, weighted_input_LBD, attn_mask)
        attention_output_BLD = attention_output_LBD.permute(1, 0, 2) # Revert back to (B, L, D) for compatibility
        attention_residual_BLD = self.norm1(attention_output_BLD + weighted_input_BLD)
        attention_dropped_out_residual_BLD = self.dropout(attention_residual_BLD)
        ff_output_BLD = self.feed_forward(attention_dropped_out_residual_BLD)
        ff_residual_BLD = self.norm2(ff_output_BLD + attention_dropped_out_residual_BLD)
        output_BLD = self.dropout(ff_residual_BLD)
        return output_BLD
        

class AtlasModelForExpressionPrediction(nn.Module):
    """
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
    def __init__(self, vocab_size, embed_dim, num_layers=6):
        super(AtlasModelForExpressionPrediction, self).__init__()
        self.gene_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.encoders = nn.ModuleList([AtlasEncoderEP() for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, input_ids_BL, input_gene_counts_BL, attn_mask=None):
        gene_embeddings_BLD = self.gene_embeddings[input_ids_BL]
        weighted_x_BLD = gene_embeddings_BLD * input_gene_counts_BL.unsqueeze(-1)
        for encoder in self.encoders:    
            weighted_x_BLD = encoder(weighted_x_BLD, attn_mask)
        output_BL = self.fc_out(weighted_x_BLD)
        return output_BL

