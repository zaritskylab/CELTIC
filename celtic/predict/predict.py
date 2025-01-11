from celtic.celtic_net.data.customizedtiffdataset import CustomizedTiffDataset
from celtic.celtic_net.fnet_model import Model
from celtic.celtic_net.transforms import normalize, normalize_with_mask, Propper
from celtic.predict.predict_functions import *
from celtic.utils.functions import load_metadata_files_and_save_locally
import os
import torch
import json
import argparse
from tqdm import tqdm

def main(args=None):
    if args is None:
        parsed_args  = vars(parse_args()) # Convert Namespace to dictionary
    elif isinstance(args, dict):
        parsed_args = args
    run_prediction(**parsed_args)

def run_prediction(path_images_csv,
                   masked,
                   transforms,
                   path_model,
                   path_context_csv,
                   path_single_cells,
                   context_features, 
                   daft_embedding_factor, 
                   daft_scale_activation,  
                   path_save_dir,
                   save_only_prediction,
                   selected_indices,
                   evaluation_metrics):
    """
    Runs the prediction pipeline on the dataset, using the specified model, transformation,
    and context information. Saves the resulting predictions and evaluation metrics.
    """

    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    if torch.cuda.is_available():
        gpu_ids = torch.cuda.current_device()
    else:
        gpu_ids = -1


    image_df, context_df = load_metadata_files_and_save_locally(['test'], path_images_csv, path_context_csv, path_save_dir, path_single_cells)
    X = image_df[0]

    if context_df:
        tabular_context_data= context_df[0]
        if len(tabular_context_data)!=len(X):
            raise RuntimeError(f'Mismatch between the number of images {len(X)} to the number of contexts {len(tabular_context_data)}')
        context_features_len = tabular_context_data.shape[1] # #TODO: consider taking from another place
        context = {'context_features': context_features,
                   'context_features_len': context_features_len, 
                   'daft_embedding_factor': daft_embedding_factor, 
                   'daft_scale_activation': daft_scale_activation}
    else:
        tabular_context_data = None
        context = None

    # load dataset
    transforms = {k: eval(v) for k, v in transforms.items()}
    dataset = CustomizedTiffDataset(dataframe = X, transforms = transforms, tabular_context_data = tabular_context_data, signals_are_masked = masked)    
    dataloader_index_map = dataset.get_index_map()
    
    if selected_indices:
        indices = selected_indices
    else:
        indices = range(len(dataset))

    # load model
    model = Model(context=context, signals_are_masked = masked)
    model.load_state(path_model, gpu_ids=gpu_ids)    
    model_str = str(model)
    model_iteration = model_str[::-1][:model_str[::-1].find(' ')][::-1]
    
    entries = []

    for idx in tqdm(indices):

        # get images
        entry = get_prediction_entry(dataset, idx)
        data = dataset[idx]
        data = [torch.unsqueeze(d, 0) for d in data] # make batch of size 1
        signal = data[dataloader_index_map['signal']]
        target = data[dataloader_index_map['target']]
        if 'mask' in dataloader_index_map:   
            mask = data[dataloader_index_map['mask']]
        else:
            mask=None

        # get tabular data
        if 'tabular_context_signal' in dataloader_index_map:
            signal_tab = data[dataloader_index_map['tabular_context_signal']]
        else:
            signal_tab = None

        # save images
        path_tiff_dir = os.path.join(path_save_dir, '{:04d}'.format(idx))
        if not save_only_prediction:
            save_tiff_and_log('signal', signal.numpy()[0, ], path_tiff_dir, entry, path_save_dir)
            save_tiff_and_log('target', target.numpy()[0, ], path_tiff_dir, entry, path_save_dir)
        if masked:   
            save_tiff_and_log('mask', mask.numpy()[0, ], path_tiff_dir, entry, path_save_dir)

        signal = signal[None, :, :, :, :]

        # predict and save
        prediction = model.predict(signal, signal_tab)
        save_tiff_and_log('prediction', prediction.numpy()[0, ], path_tiff_dir, entry, path_save_dir)
       
        entry['model_iteration'] = model_iteration
        entries.append(entry)

    if evaluation_metrics:
        evaluate(evaluation_metrics, path_save_dir, masked)

    # save args
    args = {'context_features': context_features,
            'daft_embedding_factor': daft_embedding_factor,
            'daft_scale_activation': daft_scale_activation,
            'masked': masked,
            'path_context_csv': path_context_csv,
            'path_images_csv': path_images_csv,
            'selected_indices': selected_indices
            }
    with open(f'{path_save_dir}{os.sep}config.json', "w") as file:
        file.write(json.dumps(args))

def parse_args(args=None):
                
    parser = argparse.ArgumentParser(description='Predict images on a trained model.')
    parser.add_argument('--path_images_csv', type=str, help='The file path to the images metadata CSV.')
    parser.add_argument('--masked', type=bool, help='If the provided signal and target are masked around the cell.')
    parser.add_argument('--transforms', type=dict, help='Specifies transformations for each image type (signal, target, mask) as key-value pairs.')
    parser.add_argument('--path_model', type=str, help='The file path to the trained model.')
    parser.add_argument('--path_context_csv', type=str, help='The file path to the context CSV. Provide an empty string if no context.')
    parser.add_argument('--path_single_cells', type=str, help='The directory path of the single cell images.')
    parser.add_argument('--context_features', type=list, help="List of context feature names used in the trained model. Provide an empty list for no context.")
    parser.add_argument('--daft_embedding_factor', type=int, help='Specifies the factor used to squeeze fused features in the trained DAFT model.')
    parser.add_argument('--daft_scale_activation', type=str, help='Defines the activation function used by the DAFT scaler in the trained model.')
    parser.add_argument('--path_save_dir', type=str, help='The directory path where images will be saved.')
    parser.add_argument('--save_only_prediction', type=bool, help='If True, saves only the prediction and mask; otherwise, also saves the signal and target.')
    parser.add_argument('--selected_indices', type=list, help='Indices of records to limit predictions. Provide an empty list if selecting all.')
    parser.add_argument('--evaluation_metrics', type=list, help='List of metric names for evaluating predictions against targets; leave empty for no evaluation.')
    return parser.parse_args(args) 



if __name__ == "__main__":
    main()

