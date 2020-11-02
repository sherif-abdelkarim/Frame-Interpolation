#!/bin/bash

#activate the correct envrionment
conda activate tf-gpu

# loop over all the hdf5 datasets and train a model on it
#relises on hdf5 files only exist as datasets
for f in *hdf5; do python model.py model_params.json $f; done

#deactivate the environment
conda deactivate
