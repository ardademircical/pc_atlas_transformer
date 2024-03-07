import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def scale_and_combine_selected_populations():
    selected_populations = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
    combined_data = pd.DataFrame()

    for population in selected_populations:
        multiclass_path = f"/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/wynton_code/phenotype_multiclass_label/data_{population}_phenotype_multiclass.txt"
        count_path = f"/wynton/protected/home/fhuanglab/ardademirci/Data_Specific_Epithelial/wynton_code/nn_files/count_data/data_{population}_count.txt"
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

        combined_data = pd.concat([combined_data, merged_multiclass], ignore_index=True)


    print(f"Selected populations {selected_populations} started scaling.")
    # Extract vectors and apply scaling
    vectors_df = pd.DataFrame(combined_data['vector'].tolist())
    scaler = StandardScaler()
    scaled_vectors_df = pd.DataFrame(scaler.fit_transform(vectors_df))

    # Update the 'scaled_vector' in combined_data
    combined_data['scaled_vector'] = scaled_vectors_df.values.tolist()
    combined_data['scaled_vector'] = combined_data['scaled_vector'].apply(lambda x: x[:-1])

    # Convert to 32-bit floats
    combined_data['scaled_vector'] = combined_data['scaled_vector'].apply(lambda arr: np.array(arr, dtype=np.float32).tolist())

    # Save the combined and scaled data
    scaled_combined_path = "multiclass_fp32/scaled_all_combined_data.txt"
    combined_data.to_csv(scaled_combined_path, index=False)

    print("Scaling and combining done for selected populations.")

    return combined_data

if __name__ == "__main__":
    combined_scaled_data = scale_and_combine_selected_populations()


