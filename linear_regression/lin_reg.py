import tensorflow as tf

import keras
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
#from ssim_loss import *

from keras.initializers import Initializer, Constant
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="2"


class AlphaInit(Initializer):
    def __init__(self, alpha, **kwargs):
        super(AlphaInit, self).__init__(**kwargs)
        self.constant = alpha
    def __call__(self):
        return K.constant(self.constant)

class Mult(layers.Layer):
    def __init__(self, init, **kwargs):
        super(Mult, self).__init__(**kwargs)
        self.init = init

    def build(self, input_shape):
        output_dim = input_shape
        if self.init == "constant":
            self.kernel = self.add_weight(
                shape=[1],
                initializer=Constant(0.5),
                name="kernel",
                trainable=True,
            )
        else:
            self.kernel = self.add_weight(
                shape=[1],
                initializer=keras.initializers.get("he_normal"),
                name="kernel",
                trainable=True,
            )
    def call(self, inputs):
        out = inputs*self.kernel
        return out

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        config = super(Mult, self).get_config()
        config.update({"init": self.init})
        return dict(list(config.items()))


def build_model(ssim=False):
    """
    Build Keras model with different hyperparameters
    """
    
    input_prev = Input(shape=(48, 48, 1))

    input_next = Input(shape=(48, 48, 1))

    x_prev = Mult("he")(input_prev)
    x_next = Mult("he")(input_next)  


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

    train_file = sys.argv[1]

    model_name = "lin_reg.h5"

    f  = h5py.File(train_file, 'r')
    x_train = f['x_train']
    x_train = np.array(x_train)
    shape_train = x_train.shape[0]
    shape_train = int(shape_train/64.)*64 # ssim needs to have the same batch size for all the mini batches
    x_train = x_train[:shape_train,:,:,:]
    y_train = f['y_train']
    y_train = np.array(y_train)
    y_train = y_train[:shape_train,:,:]
    
    model = build_model()
    train(model, x_train, y_train, model_name)
