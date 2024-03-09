# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gene_expression_datasets import GeneExpressionDatasetRB
from data_utils import mask_tokens, collate_fn
from atlas_model_rank_based import AtlasModelRankBased
from tokenizer import GeneTokenizer

def start_loop():
    # Assuming you have a tokenizer instance ready
    tokenizer = GeneTokenizer(vocab_file="path_to_vocab_file")

    # Load your dataset
    train_dataset = GeneExpressionDatasetRB(gene_expressions, gene_names, tokenizer)
    valid_dataset = GeneExpressionDatasetRB(valid_gene_expressions, valid_gene_names, tokenizer)

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model, optimizer, and loss function setup
    model = AtlasModelRankBased(vocab_size=len(tokenizer.vocab))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding for loss calculation

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



def train_model(model, train_loader, valid_loader, optimizer, criterion, epochs=5):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

        # Validation step
        validate_model(model, valid_loader, criterion)

def validate_model(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))

            total_loss += loss.item()

    print(f'Validation Loss: {total_loss/len(valid_loader):.4f}')
    model.train()

# Run the training loop
train_model(model, train_loader, valid_loader, optimizer, criterion, epochs=5)