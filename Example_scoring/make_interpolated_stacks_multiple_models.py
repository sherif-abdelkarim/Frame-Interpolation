####
#Script to define do a single interpolation of all numpy stacks
####
#import dependencies
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage.util import random_noise
from keras.models import Model
#from keras.models import load_model
from tensorflow.keras.models import load_model
#from ssim_loss import ssim_loss
import sys

# define interpoaltion functions
####
# MACHINE LEARNING
####
def patch_predictions(y_hat):
    """
    Reshape model predictions (patch vectors) into frames

    Args
    y_hat: (list) of [n, 576] numpy arrays from neural network model 

    Returns
    y_pred: (numpy array) of shape [n, 24, 24]
    """

    patches_hat = [y.reshape((y.shape[0], 24, 24)) for y in y_hat]
    temp1 = np.concatenate([patches_hat[0], patches_hat[2]], axis=1)
    temp2 = np.concatenate([patches_hat[1], patches_hat[3]], axis=1)
    y_pred = np.concatenate([temp1, temp2], axis=2)

    return y_pred

def ml_interpolate_series(test_series, model):
    """
    Use neural network model to interpolate tilt series. 
    From beginning to end, n - 1 frames are added into the gaps.
    For example, if there are 50 frames in the input, the ouput
    would have 99 frames after interpolation.

    Args
    test_series: (numpy array) of shape [n, d, d]
    model: (keras) trained model

    Returns
    final_series: (numpy array) of shape [2n - 1, d, d]
    """

    # create neural network input frame stack
    n_true, d, d = test_series.shape
    n_gap = n_true - 1
    model_input = np.zeros([n_gap, d, d, 2])
    for i in range(n_gap):
        model_input[i, :, :, 0] = test_series[i, :, :]
        model_input[i, :, :, 1] = test_series[i + 1, :, :]

    # use model to predict intermediate frames
    y_hat = model.predict(model_input)
    gap_series = y_hat #changed this line to see if it works

    # insert model predictions into the gaps
    n_final = n_true + n_gap
    final_series = np.zeros([n_final, d, d])
    i_true = 0
    i_gap = 0
    for j in range(n_final):
        if j % 2 == 0:
            final_series[j, :, :] = test_series[i_true, :, :]
            i_true += 1
        else:
            final_series[j, :, :] = gap_series[i_gap, :, :]
            i_gap += 1

    return final_series

####
#LINEAR 
####

def lin_interpolate(slice1, sliceN):
    """
    Docstring placeholder
    
    
    """
    output = (slice1 + sliceN)/2
    
    return output

def lin_interpolate_series(tilt_series):
    """
    Docstring Place holder
    """
    #print(tilt_series.shape)
    n_true, d, d = tilt_series.shape #how many true slices and the dimnesions of it
    n_gap = n_true - 1 # how many slices I will generate
    
    gap_series = np.zeros((n_gap, d, d))
    #print(gap_series.shape)
    for i in range(n_gap):
        
        #print(i, i +2)
        slice0 = tilt_series[i]
        #print(slice0.shape)
        slice2 = tilt_series[i + 1]
        #print(slice2.shape)
        
        slice1 = lin_interpolate(slice0, slice2)
        #print(slice2.shape)
        #print(slice1.shape)
        gap_series[i] = slice1
        
    #print(len(originals), len(generated))
      
    #final_series = np.asarray([x for x in it.chain(*it.zip_longest(tilt_series, generated)) if x is not None])
    
        # insert model predictions into the gaps
    n_final = n_true + n_gap
    final_series = np.zeros([n_final, d, d])
    i_true = 0
    i_gap = 0
    for j in range(n_final):
        if j % 2 == 0:
            final_series[j, :, :] = tilt_series[i_true, :, :]
            i_true += 1
        else:
            final_series[j, :, :] = gap_series[i_gap, :, :]
            i_gap += 1

    
    
    
    #print(tilt_series.shape, gap_series.shape, final_series.shape)
    return final_series
    


def __main__():

    #set paths

    emd_path = pathlib.Path(sys.argv[1])
    model_path = pathlib.Path(sys.argv[2])

    #get model
    #model = sorted(model_path.glob('*h5'))[0]
    #load model
    #model = load_model(model)

    #declare the steps which the models were trained with
    steps = [2, 4, 12, 20]
    
    
    #get numpy stacks
    np_files = sorted(emd_path.glob('*/*/*.npy'))
    print(len(np_files))
    
    for step in steps:
        
        #the string to seerch for in file 
        step_str = f'_step_{step}'
        
        #string to load the correct model
        model_str = f'*_{step}_aberated_*.h5'
        
        #load the model
        model_file = sorted(model_path.glob(model_str))[0]
        model = load_model(model_file, {'ssim_loss': ssim_loss})
        print(model_file.name)
        
        for file in np_files:
            #print(file.name)
            
            if step_str in file.name:
                print(file.name)
            
                #get the name of the numpy
                in_name = file.name
                
                print(in_name)
                #get the path of the numpy_file 
                in_path = file.parent
                print(in_path)

                #generate ml file name
                ml_name = in_name[:-4] + '_ml.npy'
                #generate ml path
                ml_path = in_path / ml_name

                #generate lin file name
                lin_name = in_name[:-4] + '_lin.npy'
                #generate lin path
                lin_path = in_path / lin_name

                #generate orig file name 
                orig_name = in_name[:-4] + '_orig.npy'
                #generate orig path     
                orig_path = in_path / orig_name

                #load the stack
                stack = np.load(file)
                print(stack.shape)
                stack  = np.stack(np.stack(stack,2),2)
                #stack = stack[:, 1:49, 1:49] # i changed the size form 50,50 to 48,48
                print(stack.shape)

                #generate machine learning stack
                ml_stack = ml_interpolate_series(stack, model)

                #generate linear stack
                lin_stack = lin_interpolate_series(stack)

                #save stacks
                #ML
                np.save(ml_path, np.stack(ml_stack, 2))
                #Linear
                np.save(lin_path, np.stack(lin_stack, 2))
                #orignal
                np.save(orig_path, np.stack(stack, 2))
            
            else: pass
            
            

    return None

if __name__ == "__main__":
    __main__()
