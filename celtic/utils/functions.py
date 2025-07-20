import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import json
import pandas as pd
import gdown
import os
import re
from urllib.parse import parse_qs, urlparse

def download_resources(folder_url, local_folder_path):
    """
    Downloads contents of a shared Google Drive folder using gdown.
    Works with publicly shared folder URLs.
    
    Parameters:
    folder_url (str): URL of the shared Google Drive folder
    local_folder_path (str): Path where files should be downloaded
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Create local folder if it doesn't exist
        os.makedirs(local_folder_path, exist_ok=True)
        
        # Extract folder ID from URL
        if 'folders' in folder_url:
            folder_id = folder_url.split('folders/')[-1].split('?')[0]
        else:
            parsed = urlparse(folder_url)
            folder_id = parse_qs(parsed.query).get('id', [None])[0]
            
        if not folder_id:
            raise ValueError("Could not extract folder ID from URL")
            
        # Create the folder URL format that gdown expects
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        # Download the entire folder
        gdown.download_folder(url=folder_url, 
                            output=local_folder_path,
                            quiet=False,
                            use_cookies=False)
        
        print(f"Download completed. Files saved to: {local_folder_path}")
        return True
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

def get_cell_stages():
    return ['M0','M1M2','M3','M4M5','M6M7_complete','M6M7_single']

def initialize_experiment(organelle, experiment_type, models_dir):
    
    # set path_save_dir
    formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path_save_dir = f'./experiments/{experiment_type}/{organelle}/{formatted_time}'
    
    # load configuration
    with open(f'{models_dir}/context_model_config.json', 'r') as file:
        context_model_config = json.load(file)
        
    # embed train_patch_size when needed    
    train_patch_size = context_model_config['train_patch_size']
    transforms = context_model_config['transforms']
    transforms = {k:v.replace('train_patch_size', train_patch_size) for k,v in transforms.items()}
    context_model_config['transforms'] = transforms
    context_model_config['train_patch_size'] = eval(train_patch_size)
    
    return path_save_dir, context_model_config

def initialize_randomness(seed, use_deterministic_algorithms=False):
    """
    Initialize randomness for reproducibility with an option to enforce deterministic algorithms.

    Parameters:
    seed (int): The seed value for random number generators.
    use_deterministic_algorithms (bool): Whether to enforce deterministic algorithms. Default is False.
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior at the cost of some performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optionally enforce strict deterministic algorithms
    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

def load_metadata_files_and_save_locally(file_types, path_images_csv, path_context_csv, path_run_dir, path_single_cells):
    
    # read files to dataframes
    image_df = []
    columns_to_update = ['signal_file', 'target_file', 'mask_file']
    
    # single file path
    if type(path_images_csv)==str:
        path_images_csv = [path_images_csv]
        if path_context_csv:
            path_context_csv = [path_context_csv]
        else:
            path_context_csv = []
        
    for i in range(len(path_images_csv)):
        df = pd.read_csv(path_images_csv[i])
        df[columns_to_update] = df[columns_to_update].apply(lambda col: path_single_cells + '/' + col)
        image_df.append(df)
    context_df = [pd.read_csv(path_context_csv[i]) for i in range(len(path_context_csv))]

    # save in the run directory
    for i in range(len(file_types)):
        image_df[i].to_csv(f"{path_run_dir}/{file_types[i]}_images.csv", index=False)
        if context_df:
            context_df[i].to_csv(f"{path_run_dir}/{file_types[i]}_context.csv", index=False)
    
    return image_df, context_df

def show_images_subplots(shape, images, titles=None, figsize=(20,20), axis_off=False, cmap='viridis', origin='upper', vmin=None, vmax=None, save=None, tight_layout=False):
    
    rows, columns = shape
    
    if type(cmap)==str or cmap is None:
        cmap_list = [cmap]*len(images)
    elif type(cmap)==list and len(cmap)==shape[1]:
        cmap_list = cmap * shape[0]
    elif cmap:
        assert 0, 'wrong cmap param'

    fig = plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        ax = fig.add_subplot(rows, columns, i+1)
        if img is not None:
          plt.imshow(img, cmap=cmap_list[i], origin=origin, vmin=vmin, vmax=vmax)
        if titles:
          plt.title(titles[i])
        if axis_off:
            plt.axis('off')
        plt.grid(0)
    
    if save:
        file_name, dpi = save
        plt.savefig(file_name, dpi=dpi)
        
    if tight_layout:
        plt.tight_layout()
