from wynton_code.compute_pca import compute_pca
from wynton_code.inference import run_inference
from wynton_code.prepare_data import prepare_data
from wynton_code.train import train
import argparse

def main():

    parser = argparse.ArgumentParser(description='Train a cell classifier.')
    parser.add_argument('--train_population', type=int, required=True, 
                        help='Target data population for the model')  

    args = parser.parse_args()

    prostate_populations = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
    train_population = args.train_population

    assert train_population in prostate_populations, "Target population is not from prostate tissues."
    
    train_path = f'/multiclass_fp32/scaled_multiclass_data_{train_population}.txt'
    infer_path = '/content/drive/MyDrive/tumor_classification/multiclass_fp32/scaled_multiclass_data_2.txt'

    sampled_train_df, infer_df = prepare_data(train_path, infer_path)
    train_df, valid_df, inference_df, num_components = compute_pca(sampled_train_df, infer_df)
    model = train(train_df, valid_df, num_components)
    percentages, conf_matrices, class_accuracies = run_inference(model, inference_df, num_components)