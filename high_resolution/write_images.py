import numpy as np
import os
import h5py
import random
import sys
import json

import logging as log
log.basicConfig(level=log.DEBUG)


def poisson_noise(data, scale=500.0):
    """
    Add Poisson noise by scaling the pixel values (between 0 and 1) to intergers
    then scaling back. The scale is relative and needs to be chosen carefully.

    Note scale must be a float.
    """
    d = np.where(data<0, 0, data)
    return np.random.poisson(d * scale) / scale


def prepare_data(tilt, frame_gap, noise_levels, rep, split, target_dir, i, normal=True, residual=False, res2res=False):
    h, w, n_frames = tilt.shape
    tilt = tilt/np.max(tilt)

    all_images = []

    for index in range(frame_gap, n_frames, 1):
        first_frame_index = index-frame_gap
        second_frame_index = index
        target_index = int((second_frame_index + first_frame_index)/2)
        #assert type(target_index) == int
        
        first_frame = tilt[:, :, first_frame_index]
        second_frame = tilt[:, :, second_frame_index]
        target_frame = tilt[:, :, target_index]
        
        if normal:

            stacked = np.zeros((h, w, 2))
            stacked[:, :, 0] = first_frame
            stacked[:, :, 1] = second_frame
            target = target_frame 
        
        elif residual:
            assert not res2res
            mean_frame = (first_frame + second_frame)/2.

            target_res = target_frame - mean_frame 
        
            stacked = np.zeros((w, h, 2))
            stacked[:, :, 0] = first_frame
            stacked[:, :, 1] = second_frame
            target = target_res
        
        elif res2res:
            mean_frame = (first_frame + second_frame)/2.

            first_res = first_frame - mean_frame
            second_res = second_frame - mean_frame
            target_res = target_frame - mean_frame 
        
            stacked = np.zeros((w, h, 2))
            stacked[:, :, 0] = first_res
            stacked[:, :, 1] = second_res      
            target = target_res
        
        # adding poisson noise adn rep
        noisy_repeated_images = []
        for l in noise_levels:
            for j in range(rep):
                if l == 0:
                    image = np.zeros((h, w, 3))
                    image[:, :, 0] = stacked[:, :, 0]
                    image[:, :, 1] = target
                    image[:, :, 2] = stacked[:, :, 1]
                else:
                    noisy_stacked = poisson_noise(stacked, l)
                    image = np.zeros((h, w, 3))
                    image[:, :, 0] = noisy_stacked[:, :, 0]
                    image[:, :, 1] = target
                    image[:, :, 2] = noisy_stacked[:, :, 1]
                
                file_name = os.path.join(target_dir, "image_{}_{}_{}_{}.npy".format(str(index), str(l), str(j), str(i)))
                np.save(file_name, image)


if __name__ == "__main__":

    h_params_file = sys.argv[1]
    h_params = open(h_params_file)
    h_params = json.load(h_params)

    input_file = h_params['data_path']
    frame_gap = h_params['frame_gap']
    n_series = h_params['n_series']
    noise_levels = h_params['noise_levels']
    rep = h_params['rep']
    split = h_params['train_test_split']
    target_dir = h_params['where_to_save']
    

    big_training_array = np.load(input_file)
    log.info("data is loaded")
    chunks = np.array_split(big_training_array, n_series, axis=-1)
    for i, tilt in enumerate(chunks):
        prepare_data(tilt, frame_gap, noise_levels, rep, split, target_dir, i)
        log.info("data for chunk {} are created.".format(str(i)))
