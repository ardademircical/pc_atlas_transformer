# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

from genes import PCA_GENES
import numpy as np

class GeneTokenizer:
    def __init__(self, pad_token="<PAD>", mask_token="<MASK>", unknown_token="<UNK>"):
        self.valid_genes = PCA_GENES
        self.token2idx = {pad_token: 0, mask_token: 1, unknown_token: 2}
        self.token2idx.update({gene: idx + 3 for idx, gene in enumerate(self.valid_genes)})
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unknown_token = unknown_token
        self.pad_token_id = self.token2idx[pad_token]
        self.mask_token_id = self.token2idx[mask_token]
        self.unknown_token_id = self.token2idx[unknown_token]

    def tokenize(self, gene_names, gene_expressions, use_unknown=False):
        clean_names, clean_expressions = self._remove_zeros(gene_names, gene_expressions)
        ranked_names, ranked_expressions = self._rank_genes(clean_names, clean_expressions)
        valid_names, valid_expressions = self._filter_invalid_genes(ranked_names, ranked_expressions, use_unknown)
        tokenized_genes = self._convert_to_token(valid_names, use_unknown)
        return tokenized_genes, valid_expressions
    
    def detokenize(self, token_ids):
        return [self.idx2token.get(idx, self.unknown_token) for idx in token_ids]

    ##################### Helper functions #####################
    
    def _remove_zeros(self, gene_names, gene_expressions):
        nonzero_indices = np.nonzero(gene_expressions)[0]
        return [gene_names[i] for i in nonzero_indices], gene_expressions[nonzero_indices]

    def _rank_genes(self, gene_names, gene_expressions):
        ranked_indices = np.argsort(-gene_expressions)
        return [gene_names[i] for i in ranked_indices], gene_expressions[ranked_indices]

    def _filter_invalid_genes(self, gene_names, gene_expressions, use_unknown):
        if use_unknown:
            valid_names = [name if name in self.valid_genes else "<UNK>" for name in gene_names]
            return valid_names, gene_expressions
        else:
            valid_indices = [i for i, name in enumerate(gene_names) if name in self.valid_genes]
            return [gene_names[i] for i in valid_indices], gene_expressions[valid_indices]

    def _convert_to_token(self, gene_names, use_unknown):
        if use_unknown:
            return [self.token2idx.get(name, self.unknown_token_id) for name in gene_names]
        return [self.token2idx[name] for name in gene_names if name in self.token2idx]

