import warnings
warnings.filterwarnings('ignore')

import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from global_vars import *
from preprocess import TextPreProcessor
from train import build_model

def set_seed(seed): 
    print(f'Setting SEED = {SEED} to maintain Randomness..')
    
    # Set the seed for generating random numbers in PyTorch
    torch.manual_seed(seed)
    
    # If using GPU, set the seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure that the PyTorch operations are deterministic on the GPU for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set the seed for generating random numbers in Python
    random.seed(seed)
    
    # Set the seed for generating random numbers in NumPy
    np.random.seed(seed)

set_seed(SEED)




def load_data(dataset_dict, BATCH_SIZE):
    
    print(f"\nStarting Loading and PreProcessing.... {dataset_dict['name']} \n")
    
    if dataset_dict['name'] == "tagore" :
        text_processor = TextPreProcessor(tagore_data['path'])
        dataset, total_words, longest_sequence = text_processor.create_dataset()

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = int(VALID_SPLIT * len(dataset))
    test_size = len(dataset) - (train_size + val_size)
    
    print("\nSpliting Dataset into Train, Valid, Test ratio --> {TRAIN_SPLIT}, {VALID_SPLIT}, {TEST_SPLIT}")
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f'Train Data Size -> {train_size}\tValidation Data Size -> {val_size}\tTest Data Size -> {test_size}')

    # Data Loaders
    print("\nCreating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    return total_words, longest_sequence, train_loader, valid_loader, test_loader




def main(dataset, arch_dict, batch=64, lr=0.001, epoch=100):
    
    total_words, longest_sequence, train_loader, valid_loader, test_loader = load_data(dataset, batch)

    for item in classifier_list:
        arch_name = arch_dict["name"]
        model = arch_dict['arch'](total_words, item, embedding_dim=256, hidden_dim=256, num_layers=3, dropout_rate=0.5)
        model_dict = dict(name=f"{arch_name}_{item}", arch= model, folder=f"{arch_dict['folder']}/{item}")
        
        
        model, results = build_model(model_dict, dataset, train_loader, valid_loader, test_loader, batch_size=batch, EPOCHS=epoch, LEARNING_RATE=lr)

        results = [round(x, 4) for x in results]  # formatting into 4 decimal places.


        for i in range(len(results)):

            result = results[i]
            csv_file_name = f"./{dataset['folder']}/{dataset['name']}_{csv_list[i]}"
            if os.path.exists(csv_file_name):
                df = pd.read_csv(csv_file_name)
            else:
                # Create an empty DataFrame with specified columns
                df = pd.DataFrame(columns=['name', 'fc', 'ekan'])
                df.to_csv(csv_file_name, index=False)
           
            # Update the DataFrame
            if arch_dict["name"] in df['name'].values:
                row_index = df.index[df['name'] == arch_dict['name']].tolist()[0]
                if item in df.columns:
                    df.at[row_index, item] = result
                else:
                    print(f"\nClassifier type {result} not found in columns.\n")
            else:
                # Add a new row with the architecture name and the result for the specified item
                print(f"\nArchitecture {arch_dict['name']} not found in rows... \nCreating a row with name {arch_dict['name']}..\n")
                new_row = {col: arch_dict['name'] if col == 'name' else (result if col == item else None) for col in df.columns}
                df = df._append(new_row, ignore_index=True)

            # Save the updated DataFrame back to the CSV file
            df.to_csv(csv_file_name, index=False)
                
  




  
