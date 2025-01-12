import os
import torch
import pandas as pd
from celtic.celtic_net.fnet_model import Model
from celtic.celtic_net.fnetlogger import FnetLogger
from celtic.celtic_net.data.customizedtiffdataset import CustomizedTiffDataset
from celtic.celtic_net.data import BufferedPatchDataset
from celtic.celtic_net.transforms import normalize, normalize_with_mask, Propper

def get_patch_dataloader(dataframe, 
                         tabular_context_data, 
                         transforms, 
                         masked, 
                         patch_size,
                         buffer_size,
                         buffer_switch_frequency,
                         batch_size,
                         remaining_iterations,
                         mask_efficiency_threshold, 
                         validation=False, 
                         npatches_validation=None,
                         verbose=True):
    """
    Initializes a dataloader that provides random patches from a dataset, optionally with validation support.
    """

    ds = CustomizedTiffDataset(dataframe = dataframe, transforms = transforms, tabular_context_data = tabular_context_data, signals_are_masked = masked)  
   
    # ds_patch manages the image queue. In every moment holds buffer_size images in memory.
    # moreover it has the get_random_patch method that provides a random patch upon demand.
    ds_patch = BufferedPatchDataset(
        dataset = ds,
        patch_size = patch_size,
        buffer_size = buffer_size if not validation else len(ds),
        buffer_switch_frequency = buffer_switch_frequency if not validation else -1,
        npatches = remaining_iterations * batch_size if not validation else npatches_validation,
        verbose = verbose,
        shuffle_images = True,
        weighted = True,
        mask_efficieny_threshold = mask_efficiency_threshold
    )

    dataloader = torch.utils.data.DataLoader(
        ds_patch,
        batch_size = batch_size,
    )
    return dataloader

def unpack_dataloader_item(item, dataloader_index_map, has_context, masked):
    """
    Unpacks a dataloader item into separate components: signal, target, mask, and tabular context.
    """
    
    s_tab_context = None
    s_image_context = None
    m = None
    
    # unpack signal and target
    signal = item[0]
    target = item[1]
    
    # add channel dimension
    s = signal[:,None,:,:,:] 
    t = target[:,None,:,:,:] 
        
    # unpack tabular context
    if has_context:
        s_tab_context = item[dataloader_index_map['tabular_context_signal']]
        
    # unpack mask
    if masked:
        mask = item[dataloader_index_map['mask']]
        m = mask[:,None,:,:,:]
        
    return s, t, m, s_tab_context

def initilaize_train_loggers(path_losses_csv, path_losses_val_csv, logger):
    """
    Initializes training and validation loggers by loading historical loss data if available.
    """
    
    if os.path.exists(path_losses_csv):
        fnetlogger = FnetLogger(path_losses_csv)
        logger.info('History loaded from: {:s}'.format(path_losses_csv))
    else:
        fnetlogger = FnetLogger(columns=['num_iter', 'loss_batch'])
    
    if os.path.exists(path_losses_val_csv):
        fnetlogger_val = FnetLogger(path_losses_val_csv)
        logger.info('History loaded from: {:s}'.format(path_losses_val_csv))
    else:
        fnetlogger_val = FnetLogger(columns=['num_iter', 'loss_val', 'new_best'])

    return fnetlogger, fnetlogger_val