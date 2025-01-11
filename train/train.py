import logging
import os
import sys
import torch
import time
import pandas as pd
from celtic_net.fnet_model import Model
from utils.functions import initialize_randomness, load_metadata_files_and_save_locally
from train.train_functions import *
import argparse

def main(args=None):
    if args is None:
        parsed_args  = vars(parse_args()) # Convert Namespace to dictionary
    elif isinstance(args, dict):
        parsed_args = args
    run_training(**parsed_args)

def run_training(path_run_dir,
                 path_images_csv, 
                 path_context_csv,
                 path_single_cells,
                 masked,
                 transforms,
                 patch_size,
                 iterations,
                 batch_size,
                 learning_rate,
                 context_features, 
                 daft_embedding_factor, 
                 daft_scale_activation,
                 model_random_seed = 42,
                 buffer_size = 15,
                 buffer_switch_frequency = 120,
                 mask_efficiency_threshold = 0.001,
                 validation_interval = 100):

    if torch.cuda.is_available():
        gpu_ids = torch.cuda.current_device()
    else:
        gpu_ids = -1

    if not os.path.exists(path_run_dir):
        os.makedirs(path_run_dir)

    if len(path_images_csv)!=2 or (path_context_csv and len(path_context_csv)!=2):
        raise ValueError(f"path csv lists have to contain train and validation files.")
    
    transforms = {k: eval(v) for k, v in transforms.items()}

    # load to local variables and save in the run directory 
    image_df, context_df = load_metadata_files_and_save_locally(['train', 'valid'], path_images_csv, path_context_csv, path_run_dir, path_single_cells)
    X_train, X_valid = image_df

    if context_df:
        X_train_context, X_valid_context = context_df

        # arrange context
        context_features_len = X_train_context.shape[1] # #TODO: consider taking from another place
        context = {'context_features': context_features,
                    'context_features_len': context_features_len, 
                    'daft_embedding_factor': daft_embedding_factor, 
                    'daft_scale_activation': daft_scale_activation}
    else:
        context = None
        X_train_context = None 
        X_valid_context = None
    
    time_start = time.time()

    #Setup logging
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(path_run_dir, 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Initialize Model
    path_model = os.path.join(path_run_dir, 'model.p')
    path_best_model = os.path.join(path_run_dir, 'best_model.p')
    initialize_randomness(model_random_seed)

    if os.path.exists(path_model):
        model = Model(context=context, signals_are_masked = masked)
        model.load_state(path_model, gpu_ids=gpu_ids)    
        logger.info('model loaded from: {:s}'.format(path_model))
    else:
        model = Model(
            nn_module='fnet_nn_3d',
            lr=learning_rate,
            gpu_ids=gpu_ids,
            nn_kwargs={},
            context=context,
            signals_are_masked = masked
        )
        logger.info('Model instianted from fnet_nn_3d')
    logger.info(model)
                  
    # prepare train and validation dataloaders
    n_remaining_iterations = max(0, (iterations - model.count_iter))
    dataloader_train = get_patch_dataloader(X_train, X_train_context, transforms, masked, patch_size, buffer_size, 
                                            buffer_switch_frequency, batch_size, n_remaining_iterations, mask_efficiency_threshold)
    
    npatches_validation = 50 * batch_size # will determine the average patches per image in the validation set (e.g: 50/batch_size)
    dataloader_val = get_patch_dataloader(X_valid, X_valid_context, transforms, masked, patch_size, buffer_size,
                                          buffer_switch_frequency, batch_size, n_remaining_iterations, mask_efficiency_threshold,
                                          validation=True, npatches_validation=npatches_validation)
                     
    # Load the best run's loss
    path_losses_csv = os.path.join(path_run_dir, 'losses.csv')
    path_losses_val_csv = os.path.join(path_run_dir, 'losses_val.csv')
    fnetlogger, fnetlogger_val = initilaize_train_loggers(path_losses_csv, path_losses_val_csv, logger)
    if os.path.exists(path_losses_val_csv):
        df_losses_val = pd.read_csv(path_losses_val_csv)
        val_best = df_losses_val[df_losses_val.loss_val == df_losses_val.loss_val.min()]['loss_val'].values[0]
        best_iteration = df_losses_val[df_losses_val.loss_val == df_losses_val.loss_val.min()]['num_iter'].values[0]
    else:
        val_best = 10**6
        best_iteration = 0

    dataloader_index_map = dataloader_train.dataset.dataset.get_index_map()
    criterion_val = model.criterion_fn()

    for i, item in enumerate(dataloader_train, start=model.count_iter):
    
        s, t, m, s_tab = unpack_dataloader_item(item, dataloader_index_map, (context is not None), masked)
        
        # train
        loss_batch = model.do_train_iter(s, t, mask=m, tabular_signal=s_tab)
        fnetlogger.add({'num_iter': i + 1, 'loss_batch': loss_batch})
        print('num_iter: {:6d} | loss_batch: {:.3f}'.format(i + 1, loss_batch))

        if ((i + 1) % validation_interval == 0) or ((i + 1) == iterations):
            
            model.save_state(path_model)
            fnetlogger.to_csv(path_losses_csv)
                
            loss_val_sum = 0
        
            for _, item_val in enumerate(dataloader_val):
                
                s_v, t_v, m_v, s_tab_v = unpack_dataloader_item(item_val, dataloader_index_map, (context is not None), masked)
                pred_val = model.predict(s_v, s_tab_v)
                
                if masked:
                    # loss has to be computed only on the mask pixels
                    foreground = torch.where(m_v.flatten()!=0)         
                    loss_val_batch = criterion_val(pred_val.flatten()[foreground], t_v.flatten()[foreground]).item()
                else:
                    loss_val_batch = criterion_val(pred_val, t_v).item()
                    
                loss_val_sum += loss_val_batch
                print('loss_val_batch: {:.3f}'.format(loss_val_batch))
                
            loss_val = loss_val_sum/len(dataloader_val)
            print('loss_val: {:.3f}'.format(loss_val))
            
            improvement = loss_val < val_best
            fnetlogger_val.add({'num_iter': i + 1, 'loss_val': loss_val, 'new_best': improvement})
            fnetlogger_val.to_csv(path_losses_val_csv)
            logger.info('loss val log saved to: {:s}'.format(path_losses_val_csv))
            
            # save best model
            if improvement:
                val_best = loss_val
                best_iteration = i+1
                model.save_state(path_best_model)
                logger.info('best model {} saved'.format(i+1))
                
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
   
def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--path_run_dir', type=str, help='The directory path where training products will be saved.')
    parser.add_argument('--path_images_csv', type=list, help='Two CSV file paths for train and validation image metadata.')
    parser.add_argument('--path_context_csv', type=list, help='Two CSV file paths for train and validation context metadata, or an empty string if no context.')
    parser.add_argument('--path_single_cells', type=str, help='The directory path of the single cell images.')
    parser.add_argument('--masked', type=bool, help='If the provided signal and target are masked around the cell.')
    parser.add_argument('--transforms', type=dict, help='Specifies transformations for each image type (signal, target, mask) as key-value pairs.')
    parser.add_argument('--patch_size', type=list, help='A list specifying the patch size in pixels, in the format [z, x, y].')
    parser.add_argument('--patch_size', type=list, help='A list specifying the patch size in pixels, in the format [z, x, y].')
    parser.add_argument('--iterations', type=int, help='Number of iterations for training.')
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--context_features', type=list, default='', help="List of context feature names used in the trained model. Provide an empty list for no context.")
    parser.add_argument('--daft_embedding_factor', type=int, default='', help='Specifies the factor used to squeeze fused features in the trained DAFT model.')
    parser.add_argument('--daft_scale_activation', type=str, default='', help='Defines the activation function used by the DAFT scaler in the trained model.')
    parser.add_argument('--model_random_seed', type=int, default=42, help='Random seed for model initialization.')
    parser.add_argument('--buffer_size', type=int, default=15, help='Number of images loaded to memory.')
    parser.add_argument('--buffer_switch_frequency', type=int, default=120, help='Switch interval for dataset items.')
    parser.add_argument('--mask_efficiency_threshold', type=float, default=0.001, help='Regenerate patch if mask volume percentage is below this threshold.')
    parser.add_argument('--validation_interval', type=int, default=100, help='Interval (in iterations) for model validation.')
    return parser.parse_args(args) 

if __name__ == "__main__":
    main()

