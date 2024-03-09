import torch
from torch.nn.utils.rnn import pad_sequence

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling, focusing on gene sequences."""
    labels = inputs.clone()
    
    # Generate a mask for selecting a subset of tokens to mask
    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).bool()
    
    # Only compute loss on the masked tokens by setting the labels
    labels[~masked_indices] = -100  # Tokens not selected for masking will be ignored in the loss computation
    
    # Replace selected tokens with tokenizer.mask_token_id for MLM task
    inputs[masked_indices] = tokenizer.mask_token_id
    
    return inputs, labels


def collate_fn(batch):
    """Custom collate_fn for DataLoader to process batches."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]  # Assuming labels are already part of your dataset items

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)  # Assuming 0 is your PAD token ID
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is ignored by CrossEntropyLoss

    # Create attention masks
    attention_mask = torch.zeros(input_ids_padded.shape, dtype=torch.long)
    attention_mask[input_ids_padded != 0] = 1

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }