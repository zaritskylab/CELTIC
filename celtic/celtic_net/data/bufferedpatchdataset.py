# Modified version of the code from https://github.com/AllenCellModeling/pytorch_fnet/tree/release_1

from celtic.celtic_net.data.fnetdataset import FnetDataset
import numpy as np
import torch
from tqdm import tqdm

class BufferedPatchDataset(FnetDataset):
    """
    A dataset that buffers patches for efficient data loading with options for shuffling and weighted sampling.
    """

    def __init__(self, 
                 dataset,
                 patch_size, 
                 buffer_size = 1,
                 buffer_switch_frequency = 720, 
                 npatches = 100000,
                 verbose = False,
                 transform = None,
                 shuffle_images = True,
                 dim_squeeze = None,
                 mask_efficieny_threshold = 0.15,
                 weighted = False
    ):
        
        """
        Initializes the buffered dataset with the specified parameters and prepares the buffer.
        """

        self.counter = 0
        
        self.dataset = dataset
        self.transform = transform
        self.buffer_switch_frequency = buffer_switch_frequency
        self.npatches = npatches
        self.buffer = list()
        self.buffer_volumes = list()
        self.verbose = verbose
        self.shuffle_images = shuffle_images
        self.dim_squeeze = dim_squeeze
        self.weighted = weighted
        self.mask_efficieny_threshold = mask_efficieny_threshold
        
        self.dataset_index_map = self.dataset.get_index_map()
        self.indexes_to_patch = self.dataset.get_indexes_to_patch()
        self.signal_has_mask = self.dataset.get_signal_has_mask()
        
        if 'tabular_context_signal' in self.dataset_index_map:
            self.tabular_data_index = self.dataset_index_map['tabular_context_signal']
        else:
            self.tabular_data_index = -1
        
        print('mask_efficieny_threshold initialized:', self.mask_efficieny_threshold)
        
        assert not(buffer_switch_frequency<=0 and buffer_size!=len(dataset))
        assert not(weighted and not shuffle_images)
            
        shuffed_data_order = np.arange(0, len(dataset))
        if self.shuffle_images:
            np.random.shuffle(shuffed_data_order)
            
        pbar = tqdm(range(0, buffer_size))
                       
        self.buffer_history = list()
            
        for i in pbar:
            #convert from a torch.Size object to a list
            if self.verbose: pbar.set_description("buffering images")

            datum_index = shuffed_data_order[i]
            datum = dataset[datum_index]
            datum_shape = datum[0].shape
            
            self.buffer_history.append(datum_index)
            self.buffer.append(datum)
            self.buffer_volumes.append(np.prod(datum_shape))
            
        self.remaining_to_be_in_buffer = shuffed_data_order[i+1:]    
        self.patch_size = patch_size # nts: overriding patch_size
        self.patch_volume = np.prod(self.patch_size)
          
    def __len__(self):
        return self.npatches

    def __getitem__(self, index):
        self.counter +=1
        if (self.buffer_switch_frequency > 0) and (self.counter % self.buffer_switch_frequency == 0):
            if self.verbose: print("Inserting new item into buffer")
            self.insert_new_element_into_buffer()
        return self.get_random_patch()
                       
    def insert_new_element_into_buffer(self):
        #sample with replacement
        
        self.buffer.pop(0)
        self.buffer_volumes.pop(0)
                
        if self.shuffle_images:
            
            if len(self.remaining_to_be_in_buffer) == 0:
                self.remaining_to_be_in_buffer = np.arange(0, len(self.dataset))
                np.random.shuffle(self.remaining_to_be_in_buffer)
            
            new_datum_index = self.remaining_to_be_in_buffer[0]
            self.remaining_to_be_in_buffer = self.remaining_to_be_in_buffer[1:]
            
        else:
            new_datum_index = self.buffer_history[-1]+1
            if new_datum_index == len(self.dataset):
                new_datum_index = 0
                             
        self.buffer_history.append(new_datum_index)
        
        datum = self.dataset[new_datum_index]   
        datum_shape = datum[0].shape
        self.buffer.append(datum)
        self.buffer_volumes.append(np.prod(datum_shape))

        if self.verbose: print("Added item {0}".format(new_datum_index))
        
    def get_random_patch(self):
        """
        Returns a randomly selected patch from the buffer, ensuring that the mask efficiency threshold is met.
        """
                        
        if not self.weighted:
            buffer_index = np.random.randint(len(self.buffer))
        else:
            buffer_index = np.random.choice(len(self.buffer), p=self.buffer_volumes/np.sum(self.buffer_volumes))
        
        datum = self.buffer[buffer_index]
        datum = [datum[i] for i in self.indexes_to_patch]
        
        if self.signal_has_mask:
            mask_index = self.dataset_index_map['mask']
                           
        while True:
            
            starts = np.array([np.random.randint(0, d - p + 1) if d - p + 1 >= 1 else 0 for d, p in zip(datum[0].size(), self.patch_size)])
            ends = starts + np.array(self.patch_size)

            index = [slice(s, e) for s,e in zip(starts,ends)]
                        
            patch = [d[tuple(index)] for i,d in enumerate(datum) if i!=self.tabular_data_index]            
            
            if self.dim_squeeze is not None:
                patch = [torch.squeeze(d, self.dim_squeeze) for d in patch]
            
            # regenerate patch if the mask volume precentage is less than self.mask_efficieny_threshold 
            if self.signal_has_mask:
                mask_volume = torch.sum(patch[mask_index])
                if float(mask_volume / self.patch_volume) >= self.mask_efficieny_threshold:
                    break
            else:
                break
          
        # the returned patch has to include all elements, including the ones not patched
        indexes_not_patched = set(self.dataset_index_map.values()) - set(self.indexes_to_patch)
        for i in indexes_not_patched:
            patch.insert(i, self.buffer[buffer_index][i])
                
        return patch
    
    def get_buffer_history(self):
        return self.buffer_history
    