import json
import pandas as pd
import argparse
from compute_combined_pca import compute_combined_pca
import tqdm
from multiprocessing import Pool


def process_inference_population(args):
    pop, label_map = args

    print(f"Data prep for inference population #{pop} started.")

    inference_path = f"multiclass_fp32/scaled_multiclass_data_{pop}.txt"
    inference_df = pd.read_csv(inference_path)
    inference_df = inference_df[inference_df['phenotype'] != 'Reference_LE']
    inference_df['scaled_vector'] = inference_df['scaled_vector'].apply(convert_to_list)
    try:
        inference_df.drop(columns = "Unnamed: 0", inplace=True)
    except KeyError:
        print("Column 'Unnamed: 0' not found. Continuing without dropping column.")
        pass
    inference_df['label'] = inference_df['phenotype'].map(label_map)
    
    print(f"Data prep for inference population # {pop} complete.")

    return pop, inference_df

def prepare_combined_data(selected_populations):    
    # Grab feature names
    label_map = {'Non_Tumor': 0, 'Tumor': 1, 'Basal': 2, 'Club': 3}
    f = open('feature_names.json')
    feature_names = json.load(f)
    
    print(f"Combined data prep for populations {selected_populations} started.")

    train_path = "multiclass_fp32/scaled_all_combined_data.txt"
    train_df = pd.read_csv(train_path)

    train_df = train_df[train_df['phenotype'] != 'Reference_LE']
    train_df['scaled_vector'] = train_df['scaled_vector'].apply(convert_to_list)
    try:
        train_df.drop(columns = "Unnamed: 0", inplace=True)
    except KeyError:
        print("Column 'Unnamed: 0' not found. Continuing without dropping column.")
        pass
        
    train_df['label'] = train_df['phenotype'].map(label_map)  # Map labels to integers 
    
    print(f"Combined data prep for populations {selected_populations} complete.")

    inference_populations = [15, 16]
    # Prepare arguments for multiprocessing
    args = [(pop, label_map) for pop in inference_populations]

    # Use a Pool of workers
    inference_dfs = {}
    with Pool(4) as p:
        results = p.map(process_inference_population, args)
        for pop, inference_df in results:
            inference_dfs[pop] = inference_df
    
    return train_df, inference_dfs


def convert_to_list(commaless_array_str):
    # Remove '[' and ']', then split by space and remove empty strings
    elements = commaless_array_str.strip('[]').replace(',', ' ').split()
    return [float(x) for x in elements if x]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare training data for a cell classifier.')

    selected_populations = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

    train_df, inference_dfs = prepare_combined_data(selected_populations)
    compute_combined_pca(train_df, inference_dfs, selected_populations)