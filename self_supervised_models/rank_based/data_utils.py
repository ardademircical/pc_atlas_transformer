import torch
from torch.nn.utils.rnn import pad_sequence

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling."""
    labels = inputs.clone()
    # Mask 15% of tokens in each sequence at random.
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id  # Assuming your tokenizer has a mask_token_id attribute

    # 10% of the time, replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer.vocab), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time), keep the masked input tokens unchanged
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