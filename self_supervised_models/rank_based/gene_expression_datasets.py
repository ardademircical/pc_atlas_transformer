# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

from torch.utils.data import Dataset
import torch
import numpy as np

class GeneExpressionDatasetRB(Dataset):
    def __init__(self, gene_expressions, gene_names, tokenizer, split_ratio=None, handle_oov='prune'):
        """
        Initializes the dataset with optional splitting into training and validation sets.
        
        Parameters:
        - gene_expressions: Array-like object containing gene expression data.
        - gene_names: Array-like object containing gene names.
        - tokenizer: Tokenizer object used to tokenize gene names.
        - split_ratio: Optional float indicating the ratio of the dataset to be used as training data. If None, no split is performed.
        - handle_oov: String indicating how to handle out-of-vocabulary (OOV) genes. 'prune' to remove OOV genes, 'unknown' to replace with an unknown token.
        """
        self.tokenizer = tokenizer
        self.handle_oov = handle_oov
        self.split_ratio = split_ratio
        self.data = {'train': None, 'valid': None}

        if split_ratio is not None:
            self._prepare_split_data(gene_expressions, gene_names)
        else:
            self._prepare_data(gene_expressions, gene_names)

    def _prepare_split_data(self, gene_expressions, gene_names):
        # Perform splitting
        total_size = len(gene_expressions)
        train_size = int(self.split_ratio * total_size)
        indices = torch.randperm(total_size).tolist()
        train_indices, valid_indices = indices[:train_size], indices[train_size:]

        self.data['train'] = (gene_expressions[train_indices], gene_names[train_indices])
        self.data['valid'] = (gene_expressions[valid_indices], gene_names[valid_indices])

    def _prepare_data(self, gene_expressions, gene_names):
        # No split; use all data for training
        self.data['train'] = (gene_expressions, gene_names)

    def __getitem__(self, idx):
        raise NotImplementedError("This method is not implemented. Use dataset['train'] or dataset['valid'] to access data.")

    def __len__(self):
        raise NotImplementedError("This method is not implemented. Use len(dataset['train']) or len(dataset['valid']) to get dataset size.")

    def __getitem__(self, key):
        if key not in self.data:
            raise KeyError(f"{key} split does not exist. Available splits: {list(self.data.keys())}")
        
        gene_expressions, gene_names = self.data[key]
        return SubDataset(gene_expressions, gene_names, self.tokenizer, self.handle_oov)
    

# Define a sub-dataset class that will handle the actual data retrieval
class SplitDataset(Dataset):
    def __init__(self, gene_expressions, gene_names, tokenizer, handle_oov):
        self.gene_expressions = gene_expressions
        self.gene_names = gene_names
        self.tokenizer = tokenizer
        self.handle_oov = handle_oov
    
    def __len__(self):
        return len(self.gene_expressions)
    
    def __getitem__(self, idx):
        gene_vector = self.gene_expressions[idx]
        gene_name = self.gene_names[idx]
        use_unknown = self.handle_oov == 'unknown'
        gene_tokens = self.tokenizer.tokenize([gene_name], use_unknown=use_unknown)
        
        # Apply zero-elimination and ranking
        nonzero_mask = np.nonzero(gene_vector)[0]
        if use_unknown:
            ranked_gene_tokens = self.rank_genes(gene_vector[nonzero_mask], np.array(gene_tokens)[nonzero_mask])
        else:
            # Prune OOV genes by filtering both gene_vector and gene_tokens based on nonzero_mask
            valid_tokens = [token for i, token in enumerate(gene_tokens) if i in nonzero_mask]
            ranked_gene_tokens = self.rank_genes(gene_vector[nonzero_mask], np.array(valid_tokens))
        
        return torch.tensor(ranked_gene_tokens, dtype=torch.long)
    
    def rank_genes(self, gene_vector, gene_tokens):
        """
        Rank gene expression vector.
        """
        sorted_indices = np.argsort(-gene_vector)
        return gene_tokens[sorted_indices]