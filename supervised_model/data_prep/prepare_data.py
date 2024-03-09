import json
import pandas as pd
import argparse
from compute_pca import compute_pca
import tqdm
from multiprocessing import Pool

def process_inference_population(args):
    pop, train_population, label_map = args

    print(f"Data prep for inference population # {pop} started.")

    inference_path = f"multiclass_fp32/scaled_multiclass_data_{pop}.txt"
    inference_df = pd.read_csv(inference_path)
    inference_df = inference_df[inference_df['phenotype'] != 'Reference_LE']
    inference_df['scaled_vector'] = inference_df['scaled_vector'].apply(convert_to_list)
    inference_df.drop(columns = "Unnamed: 0", inplace=True)
    inference_df['label'] = inference_df['phenotype'].map(label_map)
    
    print(f"Data prep for inference population # {pop} complete.")

    return pop, inference_df

def prepare_data(train_population):
    prostate_populations = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

    print(f"Data prep for training population # {train_population} started.")

    # Grab feature names
    f = open('feature_names.json')
    feature_names = json.load(f)

    train_path = f"multiclass_fp32/scaled_multiclass_data_{train_population}.txt"
    train_df = pd.read_csv(train_path)

    train_df = train_df[train_df['phenotype'] != 'Reference_LE']
    train_df['scaled_vector'] = train_df['scaled_vector'].apply(convert_to_list)
    train_df.drop(columns = "Unnamed: 0", inplace=True)

    label_map = {'Non_Tumor': 0, 'Tumor': 1, 'Basal': 2, 'Club': 3}
    train_df['label'] = train_df['phenotype'].map(label_map)  # Map labels to integers 

    # Filter data (abandoned for now)
    # data_label_0 = train_df[train_df['label'] == 0].sample(min(255, len(train_df[train_df['label'] == 0])), random_state=42)
    # data_label_1 = train_df[train_df['label'] == 1].sample(min(255, len(train_df[train_df['label'] == 1])), random_state=42)
    # data_label_2 = train_df[train_df['label'] == 2].sample(min(255, len(train_df[train_df['label'] == 2])), random_state=42)
    # data_label_3 = train_df[train_df['label'] == 3].sample(min(255, len(train_df[train_df['label'] == 3])), random_state=42)

    # Concatenate the two DataFrames to create your final sample DataFrame
    # sampled_train_df = pd.concat([data_label_0, data_label_1, data_label_2, data_label_3])
    # sampled_train_df = train_df


    print(f"Data prep for training population # {train_population} complete.")

    # Prepare arguments for multiprocessing
    args = [(pop, train_population, label_map) for pop in prostate_populations if pop != train_population]

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
    parser.add_argument('--train_population', type=int, required=True, 
                          help='Target data population for the model')  

    args = parser.parse_args()

    prostate_populations = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
    train_population = args.train_population

    assert train_population in prostate_populations, "Target population is not from prostate tissues."

    train_df, inference_dfs = prepare_data(train_population)
    print("All data prep done")
    compute_pca(train_df, inference_dfs, train_population)