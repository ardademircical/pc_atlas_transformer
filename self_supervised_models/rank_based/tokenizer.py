# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

class GeneTokenizer:
    def __init__(self, gene_names, pad_token="<PAD>", mask_token="<MASK>", unknown_token="<UNK>"):
        self.token2idx = {pad_token: 0, mask_token: 1, unknown_token: 2}
        self.token2idx.update({gene: idx + 3 for idx, gene in enumerate(gene_names)})
        self.idx2token = {idx: gene for gene, idx in self.token2idx.items()}
        
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unknown_token = unknown_token
        self.pad_token_id = self.token2idx[pad_token]
        self.mask_token_id = self.token2idx[mask_token]
        self.unknown_token_id = self.token2idx[unknown_token]
    
    def tokenize(self, gene_names, use_unknown=False):
        if use_unknown:
            return [self.token2idx.get(gene, self.unknown_token_id) for gene in gene_names]
        else:
            return [self.token2idx[gene] for gene in gene_names if gene in self.token2idx]

    def detokenize(self, token_ids):
        return [self.idx2token.get(idx, self.mask_token) for idx in token_ids if idx in self.idx2token]