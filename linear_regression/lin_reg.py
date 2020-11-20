import tensorflow as tf

from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import Reshape, Concatenate, Add
from keras.models import Model
import random
import keras.layers as layers
import sherpa
import sys

import numpy as np
import os
import matplotlib.pyplot as plt
from ssim_loss import *

os.environ["CUDA_VISIBLE_DEVICES"]="2"


class Mult(layers.Layer):
    def __init__(self, initializer="he_normal", **kwargs):
        super(Mult, self).__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)

    def build(self, input_shape):
        output_dim = input_shape
        self.kernel = self.add_weight(
            shape=[1],
            initializer=self.initializer,
            name="kernel",
            trainable=True,
        )

    def call(self, inputs):
        out = inputs*self.kernel
        return out

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Mult, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))


def build_model(ssim=False):
    """
    Build Keras model with different hyperparameters

    Note that adding dense_dim substantially increases network size.
    """
    # perform convolution on input images twice
    prev_w = tf.Variable(0.5)
    next_w = tf.Variable(0.5)
    
    input_prev = Input(shape=(48, 48, 1))

    input_next = Input(shape=(48, 48, 1))

    x_prev = Mult()(input_prev)
    x_next = Mult()(input_next)  

    #x_prev = conv1(x_prev)
    #x_next = conv1(x_next)

    h = Add()([x_prev, x_next])

    
    model = Model([input_prev, input_next], h)
    # add loss function
    if ssim:
        model.compile(optimizer='adam', loss=ssim_loss, metrics=[ssim_loss, 'mse'])
    else:
        model.compile(optimizer='adam', loss='mse')

    return model

    
def train(model, x_train, y_train, model_name):
    #model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val))
    x_prev = x_train[:,:,:, 0]
    x_next = x_train[:,:,:, 1]
    
    history = model.fit([x_prev, x_next], y_train, epochs=25, batch_size=64)
    model.save(model_name)

if __name__=="__main__":
    import h5py
    # we can do grid search here
    
  
    


    train_file = sys.argv[1]

    model_name = "lin_reg.h5"

    f  = h5py.File(train_file, 'r')
    x_train = f['x_train']
    x_train = np.array(x_train)
    shape_train = x_train.shape[0]
    shape_train = int(shape_train/64.)*64 # ssim needs to have the same batch size for all the mini batches
    x_train = x_train[:shape_train,:,:,:]
    y_train = f['Y_train']
    y_train = np.array(y_train)
    y_train = y_train[:shape_train,:,:]
    
    
    
    model = build_model()
    train(model, x_train, y_train, model_name)
