import tensorflow as tf

from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers import Reshape, Concatenate
from keras.models import Model
import random

import sys
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from ssim_loss import *

from data_generator import Res2ResDataGenerator

import logging as log
log.basicConfig(level=log.DEBUG)

os.environ["CUDA_VISIBLE_DEVICES"]="2"


def build_model(f1, f2, f3, k1, k2, k3, s1, s2, s3, opt, ssim=True):
    """
    Build Keras model with different hyperparameters

    """
    # perform convolution on input images twice
    inputs = Input(shape=(512, 512, 2))
    x = inputs
   
    conv1 = Conv2D(filters=f1, kernel_size=k1, activation='relu', strides=s1, padding='same')(x)
    conv2 = Conv2D(filters=f2, kernel_size=k2, activation='relu', strides=s2, padding='same')(conv1)
    output = Conv2D(filters=f3, kernel_size=k3, activation='sigmoid', strides=s3, padding='same')(conv2)
    
    model = Model(inputs, final)
    
    # add loss function
    if ssim:
        model.compile(optimizer=opt, loss=ssim_loss, metrics=[ssim_loss, 'mse'])
    else:
        model.compile(optimizer=opt, loss='mse')

    return model

    
def train(model, train_gen, val_gen, cbks, epochs, model_name):
    history = model.fit_generator(generator=train_gen, validation_data = val_gen, epochs=epochs, callbacks=cbks,
                                  max_queue_size=3, 
                                  workers=3, 
                                  use_multiprocessing=True)
    model.save(model_name)
    
    plt.figure(figsize=(10,10))
    for key in history.history.keys():
        plt.plot(history.history[key], label='%s'%key)
    plt.title('Loss Trajectory')
    plt.xlabel('epoch')
    plt.legend(loc="training trajectories")
    plt.savefig(model_name[:-3]+".png")

def build_optimizer(opt_name, lr, m, d):
    if opt_name.lower() == "sgd":
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=m, nesterov=True, name="SGD")
    elif opt_name.lowr() == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr, name="Adam")
    else:
        raise ValueError ('Unrecognized optimizer')

    return opt

def main(model_params):
    model_name = model_params['model_name']
    
    f1 = model_params['F_1']
    f2 = model_params['F_2']
    f3 = model_params['F_3']
    k1 = model_params['K_1']
    k2 = model_params['K_2']
    k3 = model_params['K_3']
    s1 = model_params['S_1']
    s2 = model_params['S_2']
    s3 = model_params['S_3']
    
    batchsize = model_params['batchsize']
    train_file_path = model_params['train_images']
    validation_path = model_params['valid_images']

    shuffle = model_params['shuffle_at_each_epoch']
    epochs = model_params['epochs']

    opt_name = model_params['optimizer_name']
    lr = model_params['learning_rate']
    m = model_params['momentum']
    d = model_params['decay']

    ssim = model_params['ssim']

    list_ids = [filename[:-4] for filename in os.listdir(train_file_path)]
    log.info("list IDs is created.")
    
    # in case you have generated res2res data using write_image.py, you can use DataGenerator instead of Res2ResDataGenerator.
    # you can uncomment the line below and comment out the line after that. Make sure you are passing the path to the res2res data directory to the image_dir argument.
    #train_generator = DataGenerator(list_IDs=list_ids, batch_size=batchsize, image_dir=train_file_path, shuffle=True)
    train_generator = Res2ResDataGenerator(list_IDs=list_ids, batch_size=batchsize, image_dir=train_file_path, shuffle=True)
    validation_generator = None
    log.info("data generators are equipped.")


    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=0,
                                                   verbose=0,
                                                   mode="auto",
                                                   baseline=None,
                                                   restore_best_weights=False)

    cbks = [early_stopping]

    optimizer = build_optimizer(opt_name, lr, m, d)

    model = build_model(f1, f2, f3, k1, k2, k3, s1, s2, s3, optimizer)
    train(model, train_generator, validation_generator, cbks, epochs, model_name)



if __name__=="__main__":
    
    model_params = sys.argv[1]
    with open(model_params, 'rb') as f:
        model_params = json.load(f)
        sys.exit(main(model_params))


