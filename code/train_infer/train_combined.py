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
from inference_combined import run_inference
import pandas as pd
import numpy as np
import json

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def convert_to_list(commaless_array_str):
    # Remove '[' and ']', then split by space and remove empty strings
    elements = commaless_array_str.strip('[]').replace(',', ' ').split()
    return [float(x) for x in elements if x]


def load_data(train_file, valid_file):
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)
    return train_df, valid_df


def train():

    original_embed_size = 1500 # PCA component size
    embed_size = 768
    num_layers = 8
    heads = 8
    num_classes = 4
    batch_size = 32
    learning_rate = 1e-4

    hyperparameters = {"original_embed_size": original_embed_size, 
                       "embed_size": embed_size, 
                       "num_layers": num_layers,
                       "heads": heads,
                       "num_classes": num_classes, 
                       "batch_size": batch_size, 
                       "learning_rate": learning_rate}

    
    train_file = f"nn_files/train_infer_data/df/combined/train_df_labeled_{original_embed_size}.csv"
    valid_file = f"nn_files/train_infer_data/df/combined/valid_df_labeled_{original_embed_size}.csv"

    train_df, valid_df = load_data(train_file, valid_file)

    train_pca_embeddings = torch.load(f"nn_files/train_infer_data/pt/combined/train_pca_vectors_{original_embed_size}.pt")
    valid_pca_embeddings = torch.load(f"nn_files/train_infer_data/pt/combined/valid_pca_vectors_{original_embed_size}.pt")

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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Adjust learning rate as needed

    # Training Loop
    num_epochs = 10  # Adjust the number of epochs as needed

    logging.info(" Training started.")
    logging.info(" Selected populations: [1,2,3,6,7,13]")
    logging.info(f"original_embed_size: {original_embed_size}")
    logging.info(f" embed_size: {embed_size}")
    logging.info(f" batch_size: {batch_size}")
    logging.info(f" num_layers: {num_layers}")
    logging.info(f" num_heads: {heads}")
    logging.info(f" num_classes: {num_classes}")
    logging.info(f" learning_rate: {learning_rate}")


    logging.info(f"original_embed_size: {original_embed_size}")

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
        validation_results = []
        last_epoch = (epoch + 1 == num_epochs)
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

                if last_epoch:
                    sequences = batch['sequence']
                    for sequence, true_label, pred_label in zip(sequences, labels, predicted):
                        validation_results.append({
                            'name': sequence,
                            'true_label': true_label.item(),
                            'predicted_label': pred_label.item()
                        })

        valid_accuracy = 100 * valid_correct / valid_total

        if last_epoch:
            validation_results_path = f"inference_results/validation_results_combined_all_data_{original_embed_size}_{hyperparameters['learning_rate']}_new_attention.json"
            with open(validation_results_path, 'w') as f:
                json.dump(validation_results, f, indent=4)  # Added indent for better readability

        
        logging.info(f" Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%, Valid Loss: {valid_loss/len(valid_loader):.4f}, Valid Accuracy: {valid_accuracy:.2f}%")
        logging.info("")

        # wandb.log({"train_acc": train_accuracy, "train_loss": loss.item(), "valid_acc": valid_accuracy, "valid_loss": valid_loss/len(valid_loader)})
        
    model_address = f"model_checkpoints/atlas_classifier_combined_all_data_{num_layers}_{original_embed_size}_{learning_rate}_new_attention.pt"
    # Save your model
    torch.save(model.state_dict(), model_address)

    # wandb.finish()

    logging.info(" Training complete.")

    return model, hyperparameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a cell classifier.')
    model, hyperparameters = train()
    run_inference(model, hyperparameters)