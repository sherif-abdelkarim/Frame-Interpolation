import tensorflow as tf

from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import Reshape, Concatenate
from keras.models import Model
import random

import sherpa
import sys
import json

import numpy as np
import os
import matplotlib.pyplot as plt
from ssim_loss import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
"""
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0],        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
"""


def build_model(f_1, f_2, k_1, k_2, s_1, s_2, d, ssim=True):
    """
    Build Keras model with different hyperparameters

    Note that adding dense_dim substantially increases network size.
    """
    # perform convolution on input images twice
    inputs = Input(shape=(48, 48, 2))
    x = inputs
    
    x = Conv2D(filters=f_1, kernel_size=k_1,
               activation='relu', strides=s_1, padding='same')(x)
 
    x = Conv2D(filters=f_2, kernel_size=k_2,
               activation='relu', strides=s_2, padding='same')(x)

    dense_dim = d
    h = Flatten()(x)
    if dense_dim > 0:
        h = Dense(dense_dim, activation='relu')(h)

    # predict four 24 x 24 segments separately
    pixel_vectors = []
    for i in range(4):
        pixel_vectors.append(Dense(576, activation='sigmoid')(h))
    outputs = []
    for i in range(4):
        outputs.append(Reshape((24, 24))(pixel_vectors[i]))
    # concatenate predicted segments together
    top = Concatenate(axis=1)([outputs[0], outputs[1]])
    bottom = Concatenate(axis=1)([outputs[2], outputs[3]])
    final = Concatenate(axis=2)([top, bottom])

    model = Model(inputs, final)
    # add loss function
    if ssim:
        model.compile(optimizer='adam', loss=ssim_loss, metrics=[ssim_loss, 'mse'])
    else:
        model.compile(optimizer='adam', loss='mse')

    return model

    
def train(model, x_train, y_train, x_val, y_val, model_name):
    #model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val))
    history = model.fit(x_train, y_train, epochs=50, batch_size=64)
    model.save(model_name)
    
    plt.figure(figsize=(10,10))
    for key in history.history.keys():
        plt.plot(history.history[key], label='%s'%key)
    plt.title('Loss Trajectory')
    plt.xlabel('epoch')
    plt.legend(loc="training trajectories")
    plt.savefig(model_name[:-3]+".png")

if __name__=="__main__":
    import h5py
    # we can do grid search here
    
    #F_1 = [8,12,16]
    #F_2 = [16,20,24]
    #K_1 = [5,7]
    #K_2 = [3,5,7]
    #S_1 = [1,2,3]
    #S_2 = [1,2]
    #D = [640, 680, 720]
    
    # or a sample of the optimized model using sherpa
    model_params = sys.argv[1]
    with open (model_params) as f:      
        model_params = json.load(f)

    train_file = sys.argv[2]

    model_name = train_file[:-5]+"_model.h5"

    f  = h5py.File(train_file)
    x_train = f['x_train']
    x_train = np.array(x_train)
    shape_train = x_train.shape[0]
    shape_train = int(shape_train/64.)*64 # ssim needs to have the same batch size for all the mini batches
    x_train = x_train[:shape_train,:,:,:]
    y_train = f['y_train']
    y_train = np.array(y_train)
    y_train = y_train[:shape_train,:,:]
    x_val = f['x_val']
    x_val = np.array(x_val)
    y_val = f['y_val']
    y_val = np.array(y_val)
    
    f_1 = model_params['F_1']
    f_2 = model_params['F_2']
    k_1 = model_params['K_1']
    k_2 = model_params['K_2']
    s_1 = model_params['S_1']
    s_2 = model_params['S_2']
    d = model_params['D']
    
    model = build_model(f_1, f_2, k_1, k_2, s_1, s_2, d)
    train(model, x_train, y_train, x_val, y_val, model_name)
