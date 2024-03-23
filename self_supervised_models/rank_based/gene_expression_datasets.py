# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

from torch.utils.data import Dataset
import torch
import numpy as np

class GeneExpressionDatasetRB(Dataset):
    def __init__(self, gene_names, gene_expressions, split_ratio=0.2):
        """
        Initializes the dataset with optional splitting into training and validation sets.
        
        Parameters:
        - gene_names: Array-like object containing gene names.
        - gene_expressions: Array-like object containing gene expression data.
        - tokenizer: Tokenizer object used to tokenize gene names.
        - split_ratio: Optional float indicating the ratio of the dataset to be used as training data. If None, no split is performed.
        """
        
        assert len(gene_names) == len(gene_expressions), "Mismatched gene expressions and names lengths."

        self.split_ratio = split_ratio
        self.data = {'train': None, 'valid': None}
    
        self._prepare_data(gene_names, gene_expressions)

    def _prepare_data(self, gene_names, gene_expressions):
        # Perform splitting
        total_size = len(gene_expressions)
        gene_expressions = np.array(eval(gene_expressions))
        train_size = int((1.0 - self.split_ratio) * total_size)
        indices = torch.randperm(total_size).tolist()
        train_indices, valid_indices = indices[:train_size], indices[train_size:]

        self.train = SplitDataset(gene_names[train_indices], gene_expressions[train_indices])
        
        if valid_indices:
            self.valid = SplitDataset(gene_names[valid_indices], gene_expressions[valid_indices])
        else:
            self.valid = None

    def __getitem__(self, split):
        if split == 'train':
            return self.train
        elif split == 'valid':
            return self.valid
        else:
            raise KeyError("Split not recognized. Use 'train' or 'valid'.")
        
    def __repr__(self):
        return f"GeneExpressionDatasetRB({{\n    train: {self.train},\n    valid: {self.valid}\n}})"
    

# Define a sub-dataset class that will handle the actual data retrieval
class SplitDataset(Dataset):
    def __init__(self, gene_names, gene_expressions):
        self.gene_names = gene_names
        self.gene_expressions = gene_expressions
    
    def __len__(self):
        return len(self.gene_expressions)
    
    def __getitem__(self, idx):
        return {
            "gene_names": self.gene_names[idx],
            "gene_expressions": self.gene_expressions[idx]
        }
    
    def __repr__(self):
        return f"({{\n features: ['gene_names', gene_expressions], \n    num_rows: {self.__len__()}\n}})"
    
    