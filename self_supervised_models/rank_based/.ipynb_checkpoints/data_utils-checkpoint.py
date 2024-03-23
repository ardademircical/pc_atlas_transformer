import torch
from torch.nn.utils.rnn import pad_sequence

def mask_tokens_for_mlm(tokens, tokenizer, mlm_probability=0.15):
    labels = tokens.copy()  # Start with a copy of tokens as labels
    probability_matrix = torch.full((len(tokens),), mlm_probability)
    
    # Decide which tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # For MLM, non-masked tokens are not predicted, so their labels are set to -100
    labels[~masked_indices] = -100
    
    # Mask selected tokens with tokenizer.mask_token_id
    for i in range(len(tokens)):
        if masked_indices[i]:
            tokens[i] = tokenizer.mask_token_id
    
    return labels, tokens

def collate_fn(batch, tokenizer, mlm_probability=0.15):
    tokenized_gene_names = [tokenizer.tokenize(item['gene_name']) for item in batch]
    max_seq_len = max(len(tokens) for tokens in tokenized_gene_names)

    # Initialize containers
    padded_gene_expressions = []
    attn_masks = []
    labels = []

    for tokens in tokenized_gene_names:
        # Pad the tokenized gene names
        padded_tokens = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
        
        # Create attention mask
        attn_mask = [1 if token != tokenizer.pad_token_id else 0 for token in padded_tokens]
        
        # Prepare labels and mask tokens for MLM
        labels_for_tokens, masked_tokens = mask_tokens_for_mlm(padded_tokens, tokenizer, mlm_probability)
        
        padded_gene_expressions.append(masked_tokens)
        attn_masks.append(attn_mask)
        labels.append(labels_for_tokens)
    
    return {
        "input_ids": torch.tensor(padded_gene_expressions, dtype=torch.long),
        "attn_mask": torch.tensor(attn_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)  # Labels prepared for MLM
    }
