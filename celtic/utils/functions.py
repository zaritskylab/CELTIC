import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import json
import pandas as pd
from pathlib import Path
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def download_file_chunked(url, dest_path, counter):
    filename = os.path.basename(url)
    destination = Path(dest_path) / filename
    if destination.exists():
        print(f"✅ Already exists: {filename}")
        return

    try:
        print(f"⬇️ Downloading [{counter}]: {filename}")
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            CHUNK = 8192
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"✅ Saved to: {destination}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

def download_example_files(resources_dir, example_type, from_dict=None):

    with open(f'{resources_dir}/example_files_config.json', 'r') as f:
        config = json.load(f)

    server_path = config["server"]
    
    if from_dict is None:
        files_to_download = {
            Path(k): v for k, v in config["files_to_download"][example_type].items()
        }
    else:
        files_to_download = {
            Path(k): v for k, v in from_dict.items()
        }
    
    # create subfolders in the resources folder
    for k in files_to_download:
        dir_path = resources_dir / k
        dir_path.mkdir(parents=True, exist_ok=True)

    # define a single download task
    def download_task(dest_dir, rel_path, counter):
        full_url = f"{server_path}/{rel_path}"
        full_dest_dir = resources_dir / dest_dir
        download_file_chunked(full_url, full_dest_dir, counter)

    # collect all download tasks
    tasks = []
    i=1
    with ThreadPoolExecutor(max_workers=4) as executor:
        for dest_dir, rel_paths in files_to_download.items():
            for rel_path in rel_paths:
                tasks.append(executor.submit(download_task, dest_dir, rel_path, i))
                i+=1

        # Optionally wait for all to finish and handle exceptions
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"Download failed: {e}")

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
