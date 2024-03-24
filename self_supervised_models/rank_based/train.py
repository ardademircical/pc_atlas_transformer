# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import StepLR
from gene_expression_datasets import GeneExpressionDatasetRB
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from data_utils import DataCollatorForGeneModeling
from atlas_model_rank_based import AtlasModelRankBased, ModelArgs
from tokenizer import GeneTokenizer
import pandas as pd
import pickle
from datasets import Dataset
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


def preprocess(tokenizer, examples):
    tokenized_output = tokenizer(examples['gene_names'], examples["gene_expressions"])
    # Return a dictionary matching the expected structure
    return {"tokenized_genes": tokenized_output}

def create_and_cache_tokenized_dataset(debug=False, tokenizer=None):
    assert tokenizer is not None, "Tokenizer must be provided"
    
    cache_prefix = "debug" if debug else "full"
    train_cache_file = f"{cache_prefix}_train_dataset.pkl"
    test_cache_file = f"{cache_prefix}_test_dataset.pkl"

    if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
        print("Loading tokenized and split datasets from cache.")
        train_dataset = Dataset.load_from_disk(train_cache_file)
        test_dataset = Dataset.load_from_disk(test_cache_file)
        return train_dataset, test_dataset

    csv_file = "/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/self_supervised_models/debug_data.csv" if debug else "/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/self_supervised_models/pc_atlas_epithelial.csv"
    print(f"Tokenizing and splitting dataset. Source: {csv_file}")

    df = pd.read_csv(csv_file)
    # gene_names is constant for all records
    with open("/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/self_supervised_models/gene_names.txt", "rb") as fp:   
        gene_names = pickle.load(fp)[:-1]
    gene_expressions = df['gene_expressions'].apply(eval).tolist()

    dataset = Dataset.from_dict({
        'gene_names': [gene_names for _ in range(len(df))],
        'gene_expressions': gene_expressions
    }).train_test_split(test_size=0.2)

    tokenized_train_set = dataset['train'].map(lambda examples: preprocess(tokenizer, examples))
    tokenized_test_set = dataset['test'].map(lambda examples: preprocess(tokenizer, examples))
    # Cache the split datasets
    tokenized_train_set.save_to_disk(train_cache_file)
    tokenized_test_set.save_to_disk(test_cache_file)
    print(f"Datasets cached: {train_cache_file}, {test_cache_file}")

    return tokenized_train_set, tokenized_test_set


def start_loop(debug=False):
    # Load your dataset
    print("Preparing dataset... \n")
    tokenizer = GeneTokenizer()
    train_dataset, test_dataset = create_and_cache_tokenized_dataset(debug=debug, tokenizer=tokenizer)
    # Setup DataLoaders
    collator_function = DataCollatorForGeneModeling(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset['tokenized_genes'], batch_size=16, shuffle=True, collate_fn=collator_function, num_workers=4)
    valid_loader = DataLoader(test_dataset['tokenized_genes'], batch_size=16, shuffle=False, collate_fn=collator_function, num_workers=4)
    print("DataLoaders setup complete. \n")

    # Model, optimizer, and loss function setup
    print("Model, optimizer, and loss function setup complete. \n")
    args = ModelArgs()
    model = AtlasModelRankBased(args)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding for loss calculation
    num_epochs = 5
    total_steps = num_epochs * len(train_loader)    
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup_steps, 
                                                num_training_steps=total_steps,
                                                num_cycles=0.5,  # Half cycle for cosine decay
                                                last_epoch=-1)

    hparam_dict = {
        'embed_dim': args.embed_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'vocab_size': 13748, # 13745 genes + <PAD> + <MASK> + <UNK>
        'norm_eps': args.norm_eps,
        'max_seq_len': args.max_seq_len,
        'dropout': args.dropout,
        'forward_expansion': 4,
        'batch_size': 16,
        'lr': 1e-4,
        'beta1': 0.9,
        'beta2': 0.98,
        'weight_decay': 0.01,
        'num_epochs': 5,
        'warmup_steps': int(0.1 * total_steps),
        'num_cycles': 0.5
    }

    # Write the hyperparameters to TensorBoard
    writer = SummaryWriter()
    writer.add_hparams(hparam_dict, {})

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Training started... \n")
    return train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, writer, num_epochs, device)

def train_model(model, train_loader, valid_loader, optimizer, scheduler, criterion, writer, num_epochs=5, device="cuda"):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler()  # Initialize the GradScaler

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, padding_mask, labels = batch['input_ids'].to(device), batch['padding_mask'].to(device), batch['labels'].to(device) 

            model.zero_grad() # Zero gradients before the forward pass

            with amp.autocast():
                outputs = model(input_ids, key_padding_mask=padding_mask) # Forward pass: Get model predictions -> (batch_size, sequence_length, vocab_size)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1)) # Labels shape: (batch_size, sequence_length)

            scaler.scale(loss).backward() # Backward pass: Compute gradient of the loss with respect to model parameters

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            
            scaler.step(optimizer) # Update parameters
            scaler.update()
            scheduler.step()  # Update learning rate
            total_loss += loss.item() # Accumulate loss for reporting

        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch: {epoch+1}, Avg Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # Validation step
        validate_model(model, valid_loader, criterion, writer, epoch, device)

    print("Training complete. \n")
    writer.close()
    # Save the model
    torch.save(model.state_dict(), 'ss_ranked_model_debug.pt')
    print("Model saved. \n")

def validate_model(model, valid_loader, criterion, writer, epoch, device="cuda"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids, padding_mask, labels = batch['input_ids'].to(device), batch['padding_mask'].to(device), batch['labels'].to(device) 
            outputs = model(input_ids, key_padding_mask=padding_mask)
            loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
            total_loss += loss.item()

    print(f'Validation Loss: {total_loss/len(valid_loader):.4f}')
    writer.add_scalar('Loss/validation', total_loss/len(valid_loader), epoch)
    print("Validation complete. \n")

# Run the training loop
if __name__ == "__main__":
    start_loop(debug=True)