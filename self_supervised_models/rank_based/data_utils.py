import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class DataCollatorForGeneModeling:
    def __init__(self, tokenizer, mlm_probability=0.15, max_seq_length=4096):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
    
    def __call__(self, batch):
        max_length = min(self.max_seq_length, max([len(seq) for seq in batch]))
        batch_input_ids = []
        batch_padding_mask = []
        batch_labels = []

        for seq in batch:
            # Trim sequences to max_seq_length
            trimmed_seq = seq[:max_length]
            input_ids, labels = self.mask_tokens(trimmed_seq, max_length)
            padding_mask = [i == self.pad_token_id for i in input_ids]
     
            batch_input_ids.append(input_ids)
            batch_padding_mask.append(padding_mask)
            batch_labels.append(labels)

        batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long)
        batch_padding_mask = torch.tensor(batch_padding_mask, dtype=torch.bool)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return {'input_ids': batch_input_ids, 'padding_mask': batch_padding_mask, 'labels': batch_labels}

    def mask_tokens(self, input_ids, max_length):
        # Create labels array
        labels = [-100] * max_length

        # Determine which tokens to mask for MLM
        probability_matrix = np.full(len(input_ids), self.mlm_probability)
        masked_indices = np.random.rand(len(input_ids)) < probability_matrix

        # Apply masking, 80% MASK, 10% random token, 10% original token
        for idx in range(len(input_ids)):
            if masked_indices[idx]:
                labels[idx] = input_ids[idx]  # Original token is the label for MLM
                if np.random.rand() < 0.8:  # 80% -> MASK
                    input_ids[idx] = self.mask_token_id
                elif np.random.rand() < 0.5:  # 10% -> Random token
                    input_ids[idx] = np.random.randint(3, len(self.tokenizer.idx2token.keys()))  # Assuming 0, 1, 2 are special tokens

        # Pad input_ids to max_length
        input_ids.extend([self.pad_token_id] * (max_length - len(input_ids)))

        return input_ids, labels
