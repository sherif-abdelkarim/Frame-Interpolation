# Frame-Interpolation
## A set up for interdisciplinary experiments and ideas for frame interpolation porject

1. Modify `data_params.json` according to your data path and run `data_generation.py` to prepare training data.
`python data_generation.py data_params.json` outputs `frame_gaps_*_aberated.hdf5`
2. Modify `model_params.json` to run `model.py`. This is a sample network.
`python model.py model_params.json frame_gaps_*_aberated.hdf5`
3. In order to evalute the trained model, you need to use eval data which was generated using `data_generation.py`. The evaluation is computing the ssim score and mse for the test data.

