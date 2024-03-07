from torch.utils.data import Dataset, DataLoader

class MulticlassGeneSplitDataset(Dataset):
    def __init__(self, sequences, labels, embeddings):
        self.sequences = sequences
        self.labels = labels
        self.embeddings = embeddings

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences.iloc[idx],
            "label": self.labels.iloc[idx],
            "embedding": self.embeddings[idx]
        }

    def __repr__(self):
        return f"Dataset({{\n    features: ['sequence', 'label', 'embedding'],\n    num_rows: {self.__len__()}\n}})"

class MulticlassGeneDataset:
    def __init__(self, train_df, train_pca_embeddings, valid_df=None, valid_pca_embeddings=None):
        self.train = MulticlassGeneSplitDataset(train_df['barcode'], train_df['label'], train_pca_embeddings)
        if valid_df is not None and valid_pca_embeddings is not None:
            self.valid = MulticlassGeneSplitDataset(valid_df['barcode'], valid_df['label'], valid_pca_embeddings)
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
        return f"DatasetDict({{\n    train: {self.train},\n    valid: {self.valid}\n}})"
