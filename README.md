# CELTIC

CELTIC (CEll in silico Labeling using Tabular Input Context) is a context-dependent model for in silico labeling of organelle fluorescence from label-free microscopy images. By incorporating biological cell contexts, CELTIC enhances the prediction of out-of-distribution data such as cells undergoing mitosis. The explicit inclusion of context has the potential to harmonize multiple datasets, paving the way for generalized in silico labeling foundation models. 

This repository contains the code, models, and data preprocessing tools for the CELTIC pipeline, as described in our [paper](https://www.biorxiv.org/content/10.1101/2024.11.10.622841v1.abstract):
<img src="assets/f2.png" width="700" />



## Overview
This repository provides the complete implementation for training, inference, and context vector generation. The structure is designed to help users easily reproduce the workflow and understand how biological context enhances organelle prediction. Below are the steps for running each part of the pipeline, along with links to the relevant Colab notebooks.
    
## Data

The datasets used for training and evaluation are available through the BioImage Archive at https://doi.org/10.6019/S-BIAD2156. The datasets contain six organelles, each with 3D single-cell images of hiPSC-derived cells with brightfield imaging, the EGFP-labeled organelle, segmentation masks, and metadata (cell cycle, edge flag, neighbors, and shape information).

The datasets can be downloaded via FTP at:
`ftp://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/156/S-BIAD2156/Files`

Each organelle has its own folder, structured as follows:


```
organelle_name/
├── cell_images/
│   ├── <FOVId_CellId>_signal.tiff
│   ├── <FOVId_CellId>_target.tiff
│   └── <FOVId_CellId>_mask.tiff
│   ```
│
├── metadata/
│   ├── metadata.csv
│   ├── context.csv
│   ├── cell_cube_coordinates_in_fov.csv
│   └── neighbours.csv


```
`cell_images`: Contains 2052–2993 3D single-cell images cropped from 180 Fields of View (FOVs). Each cell is represented by three aligned 3D images:

*   <FOVId_CellId>_signal.tiff - Brightfield
*   <FOVId_CellId>_target.tiff - EGFP-tagged organelle
*   <FOVId_CellId>_mask.tiff - Segmentation mask

`metadata.csv`: FOV and cell IDs, paths to cell images, columns from the [WTC-11 dataset](https://www.nature.com/articles/s41586-022-05563-7) (e.g., cell index in FOV mask, cell_stage).

`context.csv`: Precomputed CELTIC context for each cell (same order as metadata.csv)

`cell_cube_coordinates_in_fov.csv`: Computed cell shape descriptors.

`neighbours.csv`: Computed neighborhood features.


## Installation and Setup

1. Create a conda environment:
    ```bash
    conda create -n celtic_env python=3.9
    conda activate celtic_env
2. Clone the repository:
   ```bash
   git clone https://github.com/zaritskylab/CELTIC
   cd CELTIC
<!--
3. Install NumPy < 2.0 (NumPy 2.0 introduces breaking changes that may be incompatible with libraries such as scikit-learn, opencv-python, or older versions of PyTorch used in this project):
    ```bash
    pip install "numpy<2.0"
-->
3. Install the required dependencies:
    ```bash
    pip install .
<!--
## Running on SLURM

To train the model on a SLURM cluster, use the provided sbatch file:

```bash
sbatch train/train_celtic.sbatch
-->

## How-To Notebooks
- **Training the CELTIC Model**: 

    This notebook demonstrates how to train the CELTIC model using single cell images and context data. 
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaritskylab/CELTIC/blob/main/examples/train.ipynb)
    [![Open In Jupyter](https://img.shields.io/badge/Open%20in-Jupyter-blue.svg)](https://github.com/zaritskylab/CELTIC/blob/main/examples/train.ipynb)

    

- **Prediction with the CELTIC Model**:

    This notebook shows how to run predictions using the trained single cell model, both with and without context. It allows for the comparison of results and demonstrates how context improves the prediction accuracy.
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaritskylab/CELTIC/blob/main/examples/predict.ipynb)
    [![Open In Jupyter](https://img.shields.io/badge/Open%20in-Jupyter-blue.svg)](https://github.com/zaritskylab/CELTIC/blob/main/examples/predict.ipynb)
    
    

- **Context Creation**:

    This notebook provides a detailed walkthrough of how to create the cell context features used in the CELTIC model.
    Note that the [BioImage Archive dataset (S-BIAD2156)](https://doi.org/10.6019/S-BIAD2156) already includes precomputed context features for all single-cell images.  This notebook is useful if you want to start from scratch — for example, to take a field of view (FOV) from the [Allen Institute WTC-11 dataset](https://virtualcellmodels.cziscience.com/dataset/allencell-wtc11-hipsc-single-cell#dataset-overview), crop individual cells, and generate the corresponding context features yourself.
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zaritskylab/CELTIC/blob/main/examples/context_creation.ipynb)
    [![Open In Jupyter](https://img.shields.io/badge/Open%20in-Jupyter-blue.svg)](https://github.com/zaritskylab/CELTIC/blob/main/examples/context_creation.ipynb)  
    
## Contacts

**Author**: [Nitsan Elmalam](mailto:enitsan8@gmail.com)

**Corresponding Author**: [Assaf Zaritsky](mailto:assafzar@gmail.com)
