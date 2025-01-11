# Modified version of the code from https://github.com/AllenCellModeling/pytorch_fnet/tree/release_1

import torch.utils.data
from celtic_net.data.fnetdataset import FnetDataset
import numpy as np
import tifffile
import pandas as pd
import inspect

class CustomizedTiffDataset(FnetDataset):
    """
    Dataset for multichannel Tiff files (not supported in the original version)
    """    
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 transforms: dict,
                 tabular_context_data,
                 signals_are_masked=False):
        
        # context determines the input signal size (see set_index_map)
        # signals_are_masked: if True, the returned value will include the mask

        self.df = dataframe
        self.transform_signal = transforms['signal']
        self.transform_target = transforms['target']
        self.transform_mask = transforms['mask']
        self.tabular_context_data = tabular_context_data        
        self.signals_are_masked = signals_are_masked
        self.set_index_map()

    def set_index_map(self):
        
        self.index_map = {'signal':0, 'target':1}
        self.indexes_to_patch = [0, 1]
        
        # mask (optional)
        if self.signals_are_masked:
            self.index_map['mask'] = 2
            self.indexes_to_patch.append(2)
            
        # tabular_context_signal (optional)
        if np.any(self.tabular_context_data):
            next_index = len(self.index_map)
            self.index_map['tabular_context_signal'] = next_index
            
    def get_index_map(self):
        return self.index_map
    
    def get_indexes_to_patch(self):
        return self.indexes_to_patch
    
    def get_signal_has_mask(self):
        return self.signals_are_masked
            
    def __getitem__(self, index): 
        
        """
        Loads and transforms the signal, target, and optional mask, applying relevant transformations and returning the processed data.
        """
        
        out = list()
                
        # read the signal, target and mask
        element = self.df.iloc[index, :]
        out.append(tifffile.imread(element['signal_file']))
        out.append(tifffile.imread(element['target_file']))
        if self.signals_are_masked:
            out.append(tifffile.imread(element['mask_file']))

        imap = self.index_map
              
        # transform the signal and the target
        for t in self.transform_signal:
            if 'mask' in inspect.signature(t).parameters:
                # transfer mask when it is requested (normalize_with_mask)
                out[imap['signal']] = t(out[imap['signal']], out[imap['mask']])
            else:
                out[imap['signal']] = t(out[imap['signal']])
        for t in self.transform_target:
            if 'mask' in inspect.signature(t).parameters:
                out[imap['target']] = t(out[imap['target']], out[imap['mask']])
            else:
                out[imap['target']] = t(out[imap['target']])
            
        # optional - transform the mask
        if self.signals_are_masked:
            for t in self.transform_mask:
                out[imap['mask']] = t(out[imap['mask']])
                
        # optional - read the context data
        if np.any(self.tabular_context_data):
            out.append(self.tabular_context_data.iloc[index, :].values)
        
        # shape sanity check after padding/cropping
        signal_shapes = set([out[i].shape for i in self.indexes_to_patch])
        if len(signal_shapes)!=1:
            raise ValueError(f"Non uniform signal_shapes: {signal_shapes}")
        
        out = [torch.from_numpy(im.astype(float)).float() for im in out]
    
        return out
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index):
        return self.df.iloc[index, :].to_dict()
