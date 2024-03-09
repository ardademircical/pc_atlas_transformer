from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import pickle
import tqdm
import torch
from multiprocessing import Pool

def transform_inference_data(args):
    infer_pop, inference_df, pca, train_population = args
    print(f"PCA for inference # {infer_pop} started.")

    inference_vectors = np.stack(inference_df['scaled_vector'].values)
    inference_df['pca_vector'] = list(pca.transform(inference_vectors))

    inference_df[['barcode', 'phenotype', 'label']].to_csv(f"nn_files/train_infer_data/df/{train_population}/inference_df_labeled_{infer_pop}.csv", index=False)

    # Assuming pca_vectors is a NumPy array or a PyTorch tensor
    infer_pca_vectors = torch.tensor(np.array(inference_df['pca_vector'].tolist()))

    # Save the tensor to a file
    torch.save(infer_pca_vectors, f"nn_files/train_infer_data/pt/{train_population}/inference_pca_vectors_{infer_pop}.pt")

    print(f"PCA for inference # {infer_pop} complete.")

def compute_pca(sampled_train_df, inference_dfs, train_population):
    prostate_populations = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

    # Split the sample dataset into training and validation sets
    train_df, valid_df = train_test_split(sampled_train_df, test_size=0.2, random_state=42)

    # Prepare the data for PCA
    train_vectors = np.stack(train_df['scaled_vector'].values)
    valid_vectors = np.stack(valid_df['scaled_vector'].values)

    # Initialize PCA with the desired number of components
    if train_population == 8:
      pca = PCA(n_components=1848)
    else:
      pca = PCA(n_components=2500)


    pca.fit(train_vectors)

    # # Tentative PCA to get #PCs
    # pca_tentative = PCA().fit(train_vectors)

    # # Cumulative variance
    # cumulative_variance = np.cumsum(pca_tentative.explained_variance_ratio_)

    # Find number of components for 85% variance
    # num_components = np.argmax(cumulative_variance >= 0.85) + 1


    print(f"PCA for training population {train_population} started")

    # Save the fitted PCA model to a file
    pca_path = f"pca_models/pca_model_{train_population}.pkl"
    with open(pca_path, 'wb') as file:
        pickle.dump(pca, file)

    # Transform the training, validation, and inference data using the fitted PCA model
    train_df['pca_vector'] = list(pca.transform(train_vectors))
    valid_df['pca_vector'] = list(pca.transform(valid_vectors))

    train_df[['barcode', 'phenotype', 'label']].to_csv(f"nn_files/train_infer_data/df/{train_population}/train_df_labeled.csv", index=False)
    valid_df[['barcode', 'phenotype', 'label']].to_csv(f"nn_files/train_infer_data/df/{train_population}/valid_df_labeled.csv", index=False)

    train_pca_vectors = torch.tensor(np.array(train_df['pca_vector'].tolist()))
    valid_pca_vectors = torch.tensor(np.array(valid_df['pca_vector'].tolist()))

    # Save transformed datasets
    torch.save(train_pca_vectors, f"nn_files/train_infer_data/pt/{train_population}/train_pca_vectors_{train_population}.pt")
    torch.save(valid_pca_vectors, f"nn_files/train_infer_data/pt/{train_population}/valid_pca_vectors_{train_population}.pt")

    print("PCA for inference started")
    print("------------------------")

    # Prepare arguments for multiprocessing
    args = [(infer_pop, inference_df, pca, train_population) for infer_pop, inference_df in inference_dfs.items()]

    # Use a Pool of workers
    with Pool(4) as p:
        p.map(transform_inference_data, args)

    print("PCA for inference complete.")

    return None