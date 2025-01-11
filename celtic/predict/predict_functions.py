import numpy as np
import os
import pandas as pd
import tifffile
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def get_prediction_entry(dataset, index):
    """
    Retrieves prediction information for a given index in the dataset.
    """
    info = dataset.get_information(index)
    if isinstance(info, dict):
        return info
    if isinstance(info, str):
        return {'information': info}
    raise AttributeError

def save_tiff_and_log(tag, ar, path_tiff_dir, entry, path_log_dir, verbose=0):
    """
    Saves a NumPy array as a TIFF file and logs the file path.
    """
    if not os.path.exists(path_tiff_dir):
        os.makedirs(path_tiff_dir)
    path_tiff = os.path.join(path_tiff_dir, '{:s}.tiff'.format(tag))
    tifffile.imsave(path_tiff, ar)
    if verbose:
        print('saved:', path_tiff)
    entry['path_' + tag] = os.path.relpath(path_tiff, path_log_dir)

def evaluate(evaluation_metrics, path_save_dir, masked):
    """
    Evaluates the specified metrics for the saved images in the given directory.
    """
    df = pd.DataFrame()
    indices = [name for name in os.listdir(path_save_dir) if os.path.isdir(os.path.join(path_save_dir, name))]
    df['cell_index'] = indices
    image_file_name_format = f'{path_save_dir}{os.sep}{{}}{os.sep}{{}}.tiff'

    for metric in evaluation_metrics:
        
        if metric == 'mse':
            f = mean_squared_error
        elif metric=='pearsonr':
            f = pearsonr_statistic
        else:
            print(f"Unknown metric {metric} - skipping")
            continue
        
        results = []
        for index in indices:
            target = tifffile.imread(image_file_name_format.format(index,'target')).flatten()
            prediction = tifffile.imread(image_file_name_format.format(index,'prediction')).flatten()
            if masked:
                mask = tifffile.imread(image_file_name_format.format(index,'mask'))
                mask_indices = np.where(mask.flatten()!=0)
                target = target[mask_indices]
                prediction = prediction[mask_indices]

            results.append(f(target, prediction))
       
        df[metric] = results

    df.to_csv(f'{path_save_dir}{os.sep}evaluation.csv', index=False)

def pearsonr_statistic(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]