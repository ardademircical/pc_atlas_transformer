import logging
from multiclass_gene_dataset import MulticlassGeneDataset
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def convert_to_list(commaless_array_str):
    # Remove '[' and ']', then split by space and remove empty strings
    elements = commaless_array_str.strip('[]').replace(',', ' ').split()
    return [float(x) for x in elements if x]


def run_inference_on_single_df(model, inference_df, inference_pca_embeddings, infer_pop, device):

    label_map = {'Non_Tumor': 0, 'Tumor': 1, 'Basal': 2, 'Club': 3}
    infer_gene_dataset = MulticlassGeneDataset(inference_df, inference_pca_embeddings)
    infer_loader = DataLoader(infer_gene_dataset['train'], batch_size=32, shuffle=False)

    # Initialize confusion matrices and accuracy data for each class
    conf_matrices = {classname: {'true_labels': [], 'predicted_labels': []} for classname in label_map}
    class_accuracies = {classname: {"correct": 0, "total": 0} for classname in label_map}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    logging.info(f"Inference started on population: {infer_pop}")
    # Inference Loop
    model.eval()
    with torch.no_grad():
        for batch in infer_loader:
            embeddings = batch['embedding'].unsqueeze(1).to(device).float()
            labels = batch['label'].to(device).long()
            outputs = model(embeddings).squeeze(1)
            _, predicted = torch.max(outputs, 1)

            for true_label, pred_label in zip(labels, predicted):
                class_name = list(label_map.keys())[list(label_map.values()).index(true_label.item())]
                conf_matrices[class_name]['true_labels'].append(true_label.item())
                conf_matrices[class_name]['predicted_labels'].append(pred_label.item())
                class_accuracies[class_name]["total"] += 1
                if true_label == pred_label:
                    class_accuracies[class_name]["correct"] += 1

    percentages = {}
    for class_name in label_map:
        percentages[class_name] = class_accuracies[class_name]["correct"] / class_accuracies[class_name]["total"]

    # Print the confusion matrices and accuracy for each class
    logging.info(f" Inference results for population {infer_pop}: ")
    logging.info(percentages)

    result = {'percentages': percentages, 'conf_matrices': conf_matrices, 'class_accuracies': class_accuracies}
    return result


def run_inference(model, hyperparameters):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    inference_populations = [4, 5, 8, 12]

    # Sequential processing (more manageable on GPU memory)
    for pop in inference_populations:

        # Load the CSV file for each inference population
        inference_path = f"nn_files/train_infer_data/df/combined/inference_df_labeled_{pop}_{hyperparameters['original_embed_size']}.csv"
        inference_df = pd.read_csv(inference_path)
        inference_pca_embeddings = torch.load(f"nn_files/train_infer_data/pt/combined/inference_pca_vectors_{pop}_{hyperparameters['original_embed_size']}.pt")

        # inference_df['pca_vector'] = inference_df['pca_vector'].apply(convert_to_list)

        # Run inference on the loaded dataframe
        result = run_inference_on_single_df(model, inference_df, inference_pca_embeddings, pop, device)
        results[pop] = result

        # Offload GPU resources if any tensors were created during the inference
        torch.cuda.empty_cache()

    results_path = f"inference_results/inference_results_combined_1500_{hyperparameters['learning_rate']}_new_attention.json"
    # Assuming 'results' is the dictionary containing the results
    with open(results_path, 'w') as f:
        json.dump(results, f)
        