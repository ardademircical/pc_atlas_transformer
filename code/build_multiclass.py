import logging
import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def build_multiclass_data(population):
    phenotype_txt = f"Phenotype_Multiclass_Label/Data_{population}_phenotype_Multiclass.txt"
    train_data_txt = f"data_{population}/Data_{population}_Train_Data.txt"

    multiclass_data = pd.read_csv(phenotype_txt, sep='\t')
    train_data = pd.read_csv(train_data_txt, sep='\t')

    train_data['vector'] = train_data.apply(lambda row: row.values, axis=1)
    train_data = train_data[['vector']]
    train_data['Barcode'] = train_data.index
    train_data = train_data.reset_index(drop=True)

    df1 = multiclass_data 
    df2 = train_data

    merged_multiclass = df1.merge(df2, on='Barcode', how='inner')

    merged_multiclass = merged_multiclass.rename(columns={'Barcode': 'barcode', 'Phenotype': 'phenotype'})
    scaler = StandardScaler()
    # Extract the 'vector' column as a separate DataFrame
    vectors_df = pd.DataFrame(merged_multiclass['vector'].tolist())

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Scale the values within each array
    scaled_vectors_df = pd.DataFrame(scaler.fit_transform(vectors_df), columns=vectors_df.columns, index=vectors_df.index)

    # Add the scaled vector as a new column
    merged_multiclass['scaled_vector'] = scaled_vectors_df.values.tolist()

    vect_list = merged_multiclass['scaled_vector'].tolist()
    vect_list_np = [np.array(arr) for arr in vect_list]
    vect_list_32 = [arr.astype(np.float32) for arr in vect_list_np]
    merged_multiclass['scaled_vector'] = vect_list_32

    scaled_multiclass_data = merged_multiclass[['barcode', 'phenotype', 'scaled_vector']]
    scaled_multiclass_data['scaled_vector'] = scaled_multiclass_data['scaled_vector'].apply(list)
    file_address = f"multiclass_fp32/scaled_multiclass_data_{population}.txt"
    scaled_multiclass_data.to_csv(file_address)

    logging.info(f"Scaled multiclass dataset for population {population} is saved at {file_address}.")
    