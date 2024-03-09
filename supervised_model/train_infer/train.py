import wandb
import random
import logging
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from multiclass_gene_dataset import MulticlassGeneDataset
from atlas_transformer_multiclass import AtlasTransformerMultiClassifier
from inference import run_inference
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG)


def convert_to_list(commaless_array_str):
    # Remove '[' and ']', then split by space and remove empty strings
    elements = commaless_array_str.strip('[]').replace(',', ' ').split()
    return [float(x) for x in elements if x]


def load_data(train_file, valid_file):
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    # train_df['pca_vector'] = train_df['pca_vector'].apply(convert_to_list)
    # valid_df['pca_vector'] = valid_df['pca_vector'].apply(convert_to_list)
    return train_df, valid_df


def train(population):
    
    train_file = f"nn_files/train_infer_data/df/{population}/train_df_labeled.csv"
    valid_file = f"nn_files/train_infer_data/df/{population}/valid_df_labeled.csv"

    train_df, valid_df = load_data(train_file, valid_file)

    train_pca_embeddings = torch.load(f"nn_files/train_infer_data/pt/{population}/train_pca_vectors_{population}.pt")
    valid_pca_embeddings = torch.load(f"nn_files/train_infer_data/pt/{population}/valid_pca_vectors_{population}.pt")
    gene_dataset = MulticlassGeneDataset(train_df, train_pca_embeddings, valid_df, valid_pca_embeddings)


    # Model hyperparameters
    if population == 8:
        original_embed_size = 1848
    else:
        original_embed_size = 2500 # PCA component size

    embed_size = 768
    num_layers = 8
    heads = 8
    num_classes = 4

    # wandb.init(
    # # set the wandb project where this run will be logged
    # project= "atlas_transformer_multiclass",
    
    #     # track hyperparameters and run metadata
    #     config={
    #     "population": population,
    #     "original_embed_size": 3000,
    #     "learning_rate": 0.0001, 
    #     "weight_decay": 0.01,
    #     "architecture": "AtlasTransformerMultiClassifier",
    #     "dataset": population,
    #     "epochs": 10,
    #     "embedding_size": embed_size,
    #     "num_layers": num_layers,
    #     "heads": heads,
    #     "num_classes": num_classes,
    #     }
    # )

    # Initialize dataset
    gene_dataset = MulticlassGeneDataset(train_df, train_pca_embeddings, valid_df, valid_pca_embeddings)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AtlasTransformerMultiClassifier(original_embed_size, embed_size, num_layers, heads, device, forward_expansion=4, dropout=0.1, num_classes=num_classes)
    model.to(device)

    train_loader = DataLoader(gene_dataset['train'], batch_size=32, shuffle=True)
    valid_loader = DataLoader(gene_dataset['valid'], batch_size=32, shuffle=False)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()  # For multi-class classification, use CrossEntropyLoss
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # Adjust learning rate as needed

    # Training Loop
    num_epochs = 10  # Adjust the number of epochs as needed

    logging.info(" Training started.")
    logging.info(f" Population: {population}")
    logging.info(" ")

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            embeddings = batch['embedding'].unsqueeze(1).to(device).float()

            labels = batch['label'].to(device).long()  # Convert labels to long

            outputs = model(embeddings).squeeze(1)
            loss = criterion(outputs, labels)

            # Convert outputs to predicted labels
            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

        train_accuracy = 100 * train_correct / train_total

        # Validation Loop
        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_loss = 0

        with torch.no_grad():
            for batch in valid_loader:
                embeddings = batch['embedding'].unsqueeze(1).to(device).float()
                labels = batch['label'].to(device).long()  # Convert labels to long

                outputs = model(embeddings).squeeze(1)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        valid_accuracy = 100 * valid_correct / valid_total
        
        logging.info(f" Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss/len(valid_loader):.4f}, Valid Accuracy: {valid_accuracy:.2f}%")
        logging.info("")

        # wandb.log({"train_acc": train_accuracy, "train_loss": loss.item(), "valid_acc": valid_accuracy, "valid_loss": valid_loss/len(valid_loader)})
        
    model_address = f"model_checkpoints/atlas_classifier_{population}_{num_layers}.pt"
    # Save your model
    torch.save(model.state_dict(), model_address)

    # wandb.finish()

    logging.info(" Training complete.")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a cell classifier.')
    parser.add_argument('--train_population', type=int, required=True, 
                          help='Target data population for the model')  

    args = parser.parse_args()

    prostate_populations = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
    train_population = args.train_population

    assert train_population in prostate_populations, "Target population is not from prostate tissues."

    model = train(train_population)
    run_inference(model, train_population)
    

    