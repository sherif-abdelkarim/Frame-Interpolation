After generating the images for each frame gap, you can train models using the scripts here.
Run `patch_cnn.py params_patch_cnn.json` to train the models for patch CNN method.
Run `residual_cnn.py params_residual_cnn.json` to train the models for patch CNN method.
Run `res2res_cnn.py params_res2res_cnn.json` to train the models for patch CNN method.

The parameters in .json files are similar. For example, in `params_patch_cnn.json` you need to specify:
- F_1: Number of feature maps for the first convolution layer.
- F_2: 1 (it must be 1 for patch CNN method).
- K_1: Kernel size of the first convolution layer.
- K_2: Kernel size of the second convolution layers.
- S_1: 1 (it must be 1 for patch CNN method).
- S_2: Strides size for the second convolution layers (it must be 2 for patch CNN method).
- model_name: Name of the model to be saved. 
- batchsize: Batch size.
- train_images: `<path/to/the/directory/containing/trainig/images>`. Generated using `write_images.py`.
- valid_images: `<path/to/the/directory/containing/validation/images>`. Generated using `write_images.py`. 
- shuffle_at_each_epoch: Shuffle at the beggining of each epoch. 
- epochs: Number of epochs.
- optimizer_name: Name of the optimizer, only supports SGD and Adam.
- learning_rate: Learning rate.
- momentum: Momentum.
- decay: Decay rate.
- ssim: Whether to use SSIM loss or not.
