# Frame-Interpolation
## A set up for interdisciplinary experiments and ideas for frame interpolation porject

1. Modify `data_params.json` according to your data path and run `data_generation.py` to prepare training data.
2. Modify `model_params.json` to run `model.py`. This is a sample network.
3. In order to evalute the trained model, you need to use eval data which was generated using `data_generation.py`. The evaluation is computing the ssim score and mse for the test data.

### Residual learning

There are two methods (which are slightly different) for residual learning.
1. res2res. In order to do res2res learning, modify `data_hparams.json` and run `python res2res_data_generation.py data_hparams.json`. This will create the data you need to train a residual model by running `python model_residual.py model_params.json <path_to_the_data_generated_by_res2res_data_generation_script>`.
2. image2res. In order to do image2res learning, modify `data_hparams.json` and run `python residual_data_generation.py data_hparams.json`. This will create the data you need to train a residual model by running `python model_residual.py model_params.json <path_to_the_data_generated_by_residual_data_generation_script>`.

**NOTE 1:** For all the aforementioned cases, during the data generation, the axes of images are swapped. So you need to transpose you images during reconstruction.  
**NOTE 2:** During the data generation, the images are normalize through the entire tilt serie. This behavious has to be replicated during reconstruction.  
**NOTE 3:** In case if using residual models for reconstruction, the predicted image is supposed to be the residual map. This residual map has to be added to the linear interpolation of the two main frames to reconstuct the missing frame in the middle.  
 
