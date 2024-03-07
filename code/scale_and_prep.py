import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def scale_and_prep():
    prostate_populations = range(1, 15)
    all_scaled = {}

    # Grab feature names
    label_map = {'Non_Tumor': 0, 'Tumor': 1, 'Basal': 2, 'Club': 3}
    f = open('feature_names.json')
    feature_names = json.load(f)

    for population in prostate_populations:
        print(f"Population {population} started scaling.")
        multiclass_path = f"/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/wynton_code/phenotype_multiclass_label/data_{population}_phenotype_multiclass.txt"
        count_path = f"/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/wynton_code/nn_files/data_{population}_count.txt"
        multiclass_phenotype = pd.read_csv(multiclass_path, sep='\t')
        pop_count_data = pd.read_csv(count_path, sep='\t')
        pop_count_data = pop_count_data.T

        pop_count_data['vector'] = pop_count_data.apply(lambda row: row.values, axis=1)
        pop_count_data = pop_count_data[['vector']]
        pop_count_data['Barcode'] = pop_count_data.index
        pop_count_data = pop_count_data.reset_index(drop=True)

        df1 = multiclass_phenotype 
        df2 = pop_count_data

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

        # Get rid of the NaN at the end
        merged_multiclass['scaled_vector'] = merged_multiclass['scaled_vector'].apply(lambda x: x[:-1])

        vect_list = merged_multiclass['scaled_vector'].tolist()
        vect_list_np = [np.array(arr) for arr in vect_list]
        vect_list_32 = [arr.astype(np.float32) for arr in vect_list_np]
        merged_multiclass['scaled_vector'] = vect_list_32

        scaled_multiclass_data = merged_multiclass[['barcode', 'phenotype', 'scaled_vector']]
        scaled_multiclass_data['scaled_vector'] = scaled_multiclass_data['scaled_vector'].apply(list)

        print(f"Scaling done for population {population}.")
        print(f"Data prep for population # {train_population} started.")

        scaled_multiclass_data.drop(columns = "Unnamed: 0", inplace=True)
        scaled_multiclass_data['label'] = scaled_multiclass_data['phenotype'].map(label_map)  # Map labels to integers 

        print(f"Data prep for population # {population} complete.")

        all_scaled[population] = scaled_multiclass_data
    
    return all_scaled

if __name__ == "__main__":
    scale_and_prep()


