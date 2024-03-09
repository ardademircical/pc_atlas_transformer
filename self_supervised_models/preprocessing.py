# Author: Arda Demirci
# Email: arda.demirci@ucsf.edu

import pandas as pd
import argparse
import tqdm

def preprocess(csv_path: str):
    """
    Preprocess original prostate cancer cell atlas data.
    """

    atlas_data = pd.read_csv(csv_path)
    atlas_data = atlas_data.T
    atlas_data.rename(columns={'Unnamed: 0': 'cells'})
    atlas_data.set_index('cells', inplace=True)

    # Gather gene names
    gene_names = list(atlas_data.columns)

    # Turn expressions into array
    tqdm.pandas(desc="Calculating gene expressions")
    atlas_data['gene_expressions'] = atlas_data.iloc[:, 1:].apply(lambda row: row.tolist(), axis=1)

    # Turn names into array
    atlas_data['gene_names'] = atlas_data.apply(lambda row: gene_names, axis=1)
    
    pruned_atlas_data = atlas_data[['gene_names', 'gene_expressions']]
    pruned_atlas_data.to_hdf('pc_atlas_epithelial.h5', key='atlas', mode='w')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess prostate cancer cell atlas data.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing the data.")
    args = parser.parse_args()
    preprocess(args.csv_path)
