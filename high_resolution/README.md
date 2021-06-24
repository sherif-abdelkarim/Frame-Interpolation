Run `python write_images.py data_hparams.json` in order to generate data for training high resolution models.
Parameters you need to specify in data_hparams.json are:
- data_path: `the path to the training .npy file such as Training_ab.npy`
- n_series: 101
- n_frames: 181
- n_train: 101
- frame_gap: one integer from the list of frame gaps [2, 4, 10, 20].
- noise_levels": noise scale for the poisson noise. 0 must be included to generate the clean data.
- rep: repeatition,
- train_test_split: 1, incase of using separate train and test .npy files.
- where_to_save: `path to the directory where you want to save the images`
    * Make sure the directoy has five TB to contain the data for all the frame gaps.
    * best format is the exmaple in the data_hparams.json
- residual: if True, it will generate data for residual CNN models. Make sure to save the residual data in a separate directory than the normal data.
- res2res: if True, it will generate data for res2res CNN models. Make sure to save the res2res data in a separate directory than the normal data and residual data.
    * within the .json file, either both residual and res2res parameters are False OR only one of them is True. 
