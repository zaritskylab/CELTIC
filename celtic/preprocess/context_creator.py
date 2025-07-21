import tifffile
import pandas as pd
import numpy as np
import os
import json
import joblib
from celtic.utils.functions import get_cell_stages
from celtic.preprocess.ae import Autoencoder3D
from scipy.ndimage import zoom
import torch

class ContextVectorsCreator():
    """
    The ContextVectorsCreator class is designed to create context vectors for single cells based on their images and metadata.
    
    Attributes:
    -----------
    organelle : str
        The organelle for which the context vectors are being created.
    fovs_to_process : list
        List of Field of View (FOV) IDs to process.
    metadata_path : str
        Path to the metadata CSV file containing cell information.
    models_dir : str
        Path to the directory containing the models and configuration files.
    fov_images_dir : str
        Path to the directory containing the Field of View images.
    single_cell_image_dir : str
        Directory where single cell images will be saved.
    """
  
    def __init__(self,
                 organelle,
                 fovs_to_process,
                 metadata_path,
                 models_dir,
                 fov_images_dir,
                 single_cell_image_dir):
                
        self.organelle = organelle
        self.fovs_to_process = fovs_to_process
        self.metadata_path = metadata_path
        self.models_dir = models_dir
        self.fov_images_dir = fov_images_dir
        self.single_cell_image_dir = single_cell_image_dir
        

        # load metadata of the requested cells
        metadata = pd.read_csv(self.metadata_path)
        self.metadata = metadata[metadata.FOVId.isin(fovs_to_process)].sort_values(by=['FOVId', 'CellId'])

        # load context configuration
        with open(f'{self.models_dir}/context_model_config.json', 'r') as file:
            self.context_model_config = json.load(file)

    def create_context_vectors(self, contexts, verbose=True):
        """
        Generates context vectors by iterating over the list of context features (e.g., cell stage, shape, neighborhood) 
        and applying the respective methods. Returns a concatenated DataFrame of context vectors.
        """
        _contexts = [f'_{i}' for i in contexts]
        contexts_fn = [getattr(self, function_name, None) for function_name in _contexts]
        df_list = []
        for fn in contexts_fn:
            if verbose:
                print(fn.__name__)
            df_list.append(fn())
        return pd.concat(df_list, axis=1)

    def extract_single_cell_images(self):
        """
        Extracts single-cell images from FOVs, crops them according to the mask, and saves them in the specified directory.
        It also calculates the volume and other properties of the cell masks and stores these in a DataFrame.
        """

        os.makedirs(self.single_cell_image_dir, exist_ok=True) 
        save_path_format = self.single_cell_image_dir + '/{}_{}_{}.tiff'
        locations = []
        mask_volumes = []
        self.neighbors = []

        last_fov_id=-1
        for _, row in self.metadata.iterrows():
            
            if row.FOVId != last_fov_id:
                
                print(f"Processing FOVId {row.FOVId}")
                last_fov_id = row.FOVId

                # load signal and target
                fov_image = tifffile.imread(f'{self.fov_images_dir}/{row.fov_path}')
                signal = fov_image[:,row.ChannelNumberBrightfield]
                target = fov_image[:,row.ChannelNumberStruct]
                
                # load label map
                fov_seg_image = tifffile.imread(f'{self.fov_images_dir}/{row.fov_seg_path}')
                label_map = fov_seg_image[:, 1] # cell segmentation channel

                # collect the number of neighbors
                neighbors = self._get_number_of_neighbors(label_map, self.metadata[self.metadata.FOVId == row.FOVId].this_cell_index.tolist())
                self.neighbors.extend(neighbors)

            print(f".... CellId {row.CellId}")
            
            # signal
            cell_mask = (label_map==row.this_cell_index)
            save_path = save_path_format.format(row.FOVId, row.CellId, 'signal')
            location = save_cropped_image(signal, cell_mask, save_path)
            locations.append(location)
            mask_volumes.append(np.sum(cell_mask))
            
            # target
            save_path = save_path_format.format(row.FOVId, row.CellId, 'target')
            save_cropped_image(target, cell_mask, save_path)
            
            # mask
            save_path = save_path_format.format(row.FOVId, row.CellId, 'mask')
            label_map_binary = 1 * (label_map > 0) 
            save_cropped_image(label_map_binary.astype('uint8'), cell_mask, save_path)
            
        # coordinates are used by the classic shape context feature 
        coordinates_df = pd.DataFrame(locations, columns=['z1','z2','y1','y2','x1','x2'])
        coordinates_df.insert(0,'CellId',self.metadata.CellId.values)
        coordinates_df.insert(1,'FOVId',self.metadata.FOVId.values)
        coordinates_df['z_size'] = coordinates_df['z2']-coordinates_df['z1']+1
        coordinates_df['y_size'] = coordinates_df['y2']-coordinates_df['y1']+1
        coordinates_df['x_size'] = coordinates_df['x2']-coordinates_df['x1']+1
        coordinates_df['volume'] = coordinates_df['z_size'] * coordinates_df['y_size'] * coordinates_df['x_size']
        coordinates_df['mask_volume'] = mask_volumes
        self.coordinates = coordinates_df
    
    def _cell_stage(self):

        # encode cell stage data
        stages = {item:index for index,item in enumerate(get_cell_stages())}
        cell_stage_encoded = self.metadata.cell_stage.map(stages).values

        # transform to one-hot
        categories = np.array(list(stages.values()))
        return custom_get_dummies(cell_stage_encoded, 'cell_stage', categories)

    def _location(self):
        return self.metadata.edge_flag.reset_index(drop=True).rename('location')

    def _classic_shape(self):
        
        # load the pretrained clustering model
        kmeans = joblib.load(f'{self.models_dir}/classic_shape_kmeans.pkl')
        scaler = joblib.load(f'{self.models_dir}/classic_shape_scaler.pkl')
        categories=range(5)

        # scale
        columns_to_scale = ['z1', 'z_size', 'y_size', 'x_size', 'mask_volume']
        scaled_coordinates = pd.DataFrame(scaler.transform(self.coordinates[columns_to_scale]), columns=columns_to_scale)
        scaled_coordinates['CellId'] = self.coordinates['CellId']

        # compute row-wise min and max of y_size and x_size, indifferent to the axis
        scaled_coordinates['min_xy'] = scaled_coordinates[['y_size', 'x_size']].min(axis=1)
        scaled_coordinates['max_xy'] = scaled_coordinates[['y_size', 'x_size']].max(axis=1)

        # cluster
        columns_to_cluster = ['z_size', 'mask_volume', 'min_xy', 'max_xy']
        pred = kmeans.predict(scaled_coordinates[columns_to_cluster])

        # represent as one-hot
        return custom_get_dummies(pred, 'classic_shape', categories)
    
    def _ml_shape(self):

        # create shape images
        mask_paths = [f"{self.single_cell_image_dir}/{{}}_{{}}_mask.tiff".format(fov_id,cell_id) for fov_id, cell_id in self.metadata[['FOVId', 'CellId']].values] 
        shape_images = [self._create_cell_shape_image(tifffile.imread(path)) for path in mask_paths]

        # load pretrained autoencoder
        autoencoder = Autoencoder3D()
        autoencoder.load_state_dict(torch.load(f'{self.models_dir}/ae.pth'))
        autoencoder.eval();

        # extract encodings
        outputs = []
        for si in shape_images:
            input = torch.tensor(si, dtype=torch.float32).unsqueeze(0)
            outputs.append(autoencoder.encoder(input).cpu().detach().numpy())
        features = np.array(outputs)
        features = features.reshape(features.shape[0],-1)

        # load the pretrained pca and clustering model
        pca = joblib.load(f'{self.models_dir}/ml_shape_pca.pkl')
        kmeans = joblib.load(f'{self.models_dir}/ml_shape_kmeans.pkl')
        categories=range(3)

        # cluster
        pca_features = pca.transform(features)
        pred = kmeans.predict(pca_features)
        
        # represent as one-hot
        return custom_get_dummies(pred, 'ml_shape', categories)
        
    def _neighborhood_density(self):

        if not hasattr(self, 'neighbors'):
            raise RuntimeError("Neigbors not initialized. Call extract_single_cell_images() first.")

        df = pd.DataFrame(self.neighbors, columns = ['neighborhood_density'])

        # scale by the dataset's scaling values
        scale_min, scale_max = eval(self.context_model_config['neighbors_min_max'])
        df['neighborhood_density'] = (df['neighborhood_density'] - scale_min) / (scale_max - scale_min)

        return df

    def _get_number_of_neighbors(self, label_map, cell_indexes):

        results = []

        for this_cell_index in cell_indexes:
            
            # load cell label map indexes
            this_cell_neigbours = []
            z_indexes, y_indexes, x_indexes = np.where(label_map==this_cell_index)

            # neigbours on the z axis
            this_cell_neigbours.extend(set(label_map[z_indexes+1,y_indexes, x_indexes]))
            this_cell_neigbours.extend(set(label_map[z_indexes-1,y_indexes, x_indexes]))

            # neigbours on the y axis
            this_cell_neigbours.extend(set(label_map[z_indexes,y_indexes+1, x_indexes]))
            this_cell_neigbours.extend(set(label_map[z_indexes,y_indexes-1, x_indexes]))

            # neigbours on the x axis
            this_cell_neigbours.extend(set(label_map[z_indexes,y_indexes, x_indexes+1]))
            this_cell_neigbours.extend(set(label_map[z_indexes,y_indexes, x_indexes-1]))

            # find unique values, and remove 0 and self
            this_cell_neigbours = set(this_cell_neigbours) - set([0, this_cell_index])
            results.append(len(this_cell_neigbours))

        return results

    def _create_cell_shape_image(self, cell_mask, final_image_size = [32,64,64]):

        # create a zero-value image, padded to fit the largest cell image in the dataset
        max_coordinates = eval(self.context_model_config['max_cell_coordinates'])
        padded_shape = [int(p * np.ceil(max_coordinates[i]/p)) for i, p in enumerate(final_image_size)]
        boxed_img = np.zeros(padded_shape)
        
        # place the mask in the zero padded image
        start_indices = [padded_shape[i]//2 - cell_mask.shape[i]//2 for i in range(3)]
        slices = [slice(start_indices[i], start_indices[i] + cell_mask.shape[i]) for i in range(3)]
        boxed_img[slices[0], slices[1], slices[2]] = cell_mask
        assert len(np.where(cell_mask>0)[0]) == len(np.where(boxed_img>0)[0])

        # resize to final_image_size
        zoom_factors = [final_image_size[i] / padded_shape[i] for i in range(3)]
        final_image = zoom(boxed_img, zoom_factors, mode='nearest', order=0) 
        assert np.all(np.unique(final_image) == [0,1])

        return final_image

def save_cropped_image(image, mask, save_path, mask_value=0, normalize=False):
    """
    Crops a cell from a full FOV-sized image based on a binary mask, and optionally normalizes and pads the image. 
    The cropped image is then saved to the specified location.
    
    Parameters:
    -----------
    image : numpy.ndarray
        A 3D numpy array representing the full FOV-sized signal or target image.
    mask : numpy.ndarray
        A binary 3D numpy array of the same size as the image, with a single cell (or region) marked by 1s.
    save_path : str
        The path where the cropped image will be saved. If None, the image is not saved.
    mask_value : int, optional
        The value used to fill the background in the masked image. Default is 0.
    normalize : bool, optional
        If True, normalizes the image based on the non-masked pixels (default is False).

    Returns:
    --------
    location : list
        A list containing the minimum and maximum indices (z, y, x) of the region of interest (cell) in the image.
    """
    
    mask_indices = np.where(mask)
    
    # create a fov-sized image, with only the cell signal/target in it, and the rest is mask_value.  
    image_masked = np.where(mask, image, mask_value)
    
    if image_masked.shape[0]<=16:
        print(f"image with shape {image_masked.shape[0]} ({save_path}) padded to 17")
        box = np.ones([17, image_masked.shape[1], image_masked.shape[2]]) * mask_value
        start_z = (17 - image_masked.shape[0]) // 2
        box[start_z,:,:] = image_masked
        
    # sanity/safety check: the image does not contain 0s. this is checked by calculating the amount of pixels in the mask vs. the amount of non 0 pixels in the masked image.
    assert mask.sum()==image_masked[image_masked!=mask_value].size, f'{mask.sum()}, {image_masked[image_masked!=mask_value].size}'
    
    # normalize (only the non-mask pixels)
    if normalize:
        normalize_mean = 0
        normalize_std = 1
        non_mask_pixels = image_masked[mask]
        non_mask_pixels = non_mask_pixels.astype(np.float64)
        non_mask_mean = np.mean(non_mask_pixels)
        non_mask_std = np.std(non_mask_pixels)

        image_masked = np.where(image_masked!=mask_value, ((image_masked - non_mask_mean) / non_mask_std) * normalize_std + normalize_mean , mask_value)
    
    location = []

    # find min and max values of z,y,x
    for i in range(3):
        location.extend([np.min(mask_indices[i]),np.max(mask_indices[i])])
    
    image_cropped = image_masked[location[0]:location[1]+1, location[2]:location[3]+1, location[4]:location[5]+1]
        
    if save_path:
        tifffile.imwrite(save_path, image_cropped)
    return location

def custom_get_dummies(data, col_prefix, possible_values):
    """
    Create dummy variables for a column, ensuring all possible values are represented
    even when they're not present in the input data. 
    
    Parameters:
    -----------
    data : pandas.DataFrame or Series
        Input data containing the column to encode
    col_prefix : str
        Name of the column to encode
    possible_values : list
        List of all possible values that should be represented in the output
        
    Returns:
    --------
    pandas.DataFrame
        One-hot encoded DataFrame with columns for all possible values
    """
    
    # Create dummy variables for the actual values in the data
    dummies = pd.get_dummies(data, prefix=col_prefix, dtype=int)

    # Add missing columns with zeros
    for value in possible_values:
        
        col_name = f"{col_prefix}_{value}"
        if col_name not in dummies.columns:
            dummies[col_name] = 0
            
    # Sort columns to ensure consistent order
    return dummies.reindex(sorted(dummies.columns), axis=1)
