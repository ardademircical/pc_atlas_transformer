# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from torch.utils.tensorboard import SummaryWriter, add_hparams

def create_dataset(debug=False):
    # gene_names is constant for all records
    with open("/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/self_supervised_models/gene_names.txt", "rb") as fp:   
        gene_names = pickle.load(fp)[:-1]
    if debug:
        print("DEBUG RUN")
        df = pd.read_csv("/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/self_supervised_models/debug_data.csv")
    else:
        print("FULL TRAINING RUN")
        df = pd.read_csv("/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/self_supervised_models/pc_atlas_epithelial.csv")
    
    # Directly create a dataset without repeating gene_names for each record
    gene_expressions = [eval(exp) for exp in df['gene_expressions']]
    dataset = Dataset.from_dict({
        'gene_names': [gene_names for _ in range(len(df))],
        'gene_expressions': gene_expressions
    })
    
    dataset = dataset.train_test_split(test_size=0.2)
    print("")
    print("Dataset splits created. \n")
    return dataset

def preprocess(tokenizer, examples):
    tokenized_output = tokenizer(examples['gene_names'], examples["gene_expressions"])
    # Return a dictionary matching the expected structure
    return {"tokenized_genes": tokenized_output}

def start_loop(debug=False):
    # Load your dataset
    print("Dataset creation started... \n")
    dataset = create_dataset(debug)
    print("Dataset creation complete. \n")
    tokenizer = GeneTokenizer()
    print("Tokenization started ... \n")
    tokenized_train_set = dataset['train'].map(lambda examples: preprocess(tokenizer, examples))
    tokenized_test_set = dataset['test'].map(lambda examples: preprocess(tokenizer, examples))
    print("Tokenization complete. \n")
    collator = DataCollatorForGeneModeling(tokenizer=tokenizer)

    # DataLoader setup
    train_loader = DataLoader(tokenized_train_set['tokenized_genes'], batch_size=16, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(tokenized_test_set['tokenized_genes'], batch_size=16, shuffle=False, collate_fn=collator)
    print("DataLoader setup complete. \n")

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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, padding_mask, labels = batch['input_ids'].to(device), batch['padding_mask'].to(device), batch['labels'].to(device) 

            model.zero_grad() # Zero gradients before the forward pass

            outputs = model(input_ids, key_padding_mask=padding_mask) # Forward pass: Get model predictions

            # Model outputs logits of: (batch_size, sequence_length, vocab_size)
            # Labels shape: (batch_size, sequence_length)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            loss.backward() # Backward pass: Compute gradient of the loss with respect to model parameters

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            
            optimizer.step()  # Update parameters
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