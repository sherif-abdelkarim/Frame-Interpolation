from datetime import datetime
import tensorflow as tf
import keras
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
import cv2
import h5py

from keras.backend import int_shape
from keras.layers import (
    BatchNormalization, Conv2D, Conv2DTranspose, SpatialDropout2D,
    MaxPooling2D, Dropout, Input, concatenate, Cropping2D
)
import cv2


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#tf.config.set_visible_devices(physical_devices[3], 'GPU')
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
print('Is GPU available?', tf.test.is_gpu_available())
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])
gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    print("Type: {}, name: {}".format(gpu.name, gpu.device_type))
    tf.config.experimental.set_memory_growth(gpu, True)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, delta, noise_levels, batch_size=32, dim=(512, 512), dim_out=(324, 324), n_channels=2, shuffle=True):
        'Initialization'
        self.data = data
        self.dim = dim
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.noise_levels = noise_levels
        self.delta = delta
        self.real_list_IDs = list(range(len(data))) # [(self.delta - 1):]
        self.list_IDs = list(range(len(data))) # [(self.delta - 1):]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index = max(index, self.delta)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')

    #         # Store class
    #         y[i] = self.labels[ID]

    #     return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __data_generation(self, list_IDs_temp):
        w, h = self.dim
        w_out, h_out = self.dim_out
        x_train_np = np.empty((self.batch_size, w, h, 2))
        y_train_np = np.empty((self.batch_size, w_out, h_out, 1))

        for i, ID in enumerate(list_IDs_temp):
            index = self.real_list_IDs[ID]

            first_frame_index = index - self.delta
            second_frame_index = index

            target_index = int((second_frame_index + first_frame_index)/2.)

            x_train_np[i, :, :, 0] = self.data[first_frame_index, :, :]
            x_train_np[i, :, :, 1] = self.data[second_frame_index, :, :]

            if len(self.noise_levels) > 0:
                x_train_np = poisson_noise(x_train_np, random.choice(self.noise_levels))

            #y_train_np[i, :, :] = cv2.resize(self.data[target_index, :, :], (324, 324), interpolation=cv2.INTER_CUBIC)
            target = self.data[target_index, :, :]
            if target.shape != (w_out, h_out):
                target = cv2.resize(target, (w_out, h_out), interpolation=cv2.INTER_CUBIC)
            else:
                y_train_np[i, :, :, 0] = target 
            
        return x_train_np, y_train_np


# def data_generator(tilt_serie, delta, noise_levels):
    # n_frames, w, h = tilt_serie.shape
    # #for index in range(delta, n_frames, 1):
    # index = delta
    # i = delta
    # while i < n_frames:
    #     print('{}/{}'.format(i, n_frames))
    #     #if index == n_frames:
    #     #    index = delta
    #     index = random.randint(delta, n_frames - 1)
    #     first_frame_index = index - delta
    #     second_frame_index = index
    #     #print('tilt_serie', tilt_serie.shape,'index', index, 'first_frame_index', first_frame_index, 'target_index', target_index)
    #     target_index = int((second_frame_index + first_frame_index)/2.)
    #     # if target_index % 1000 == 0:
    #     x_train_np = np.empty((1, 512, 512, 2))
    #     y_train_np = np.empty((1, 324, 324))

    #     x_train_np[0, :, :, 0] = tilt_serie[first_frame_index, :, :]
    #     x_train_np[0, :, :, 1] = tilt_serie[second_frame_index, :, :]
    #     x_train_np = poisson_noise(x_train_np, random.choice(noise_levels))
    #     #target = tilt_serie[target_index, :, :] 
    #     # y_train_np[0, :, :] = tilt_serie[target_index, :, :]
    #     # y_train_np[0, :, :] = tilt_serie[target_index, :324, :324]
    #     y_train_np[0, :, :] = cv2.resize(tilt_serie[target_index, :, :], (324, 324), interpolation=cv2.INTER_CUBIC)
    #     # index += 1
    #     i += 1
    #     yield x_train_np, y_train_np


def poisson_noise(data, scale=500.0):
    """
    Add Poisson noise by scaling the pixel values (between 0 and 1) to intergers
    then scaling back. The scale is relative and needs to be chosen carefully.

    Note scale must be a float.
    """
    return np.random.poisson(data * scale) / scale


def conv2d_block(
    inputs,
    use_batch_norm=True,
    dropout=0.3,
    dropout_type="spatial",
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
):

    if dropout_type == "spatial":
        DO = SpatialDropout2D
    elif dropout_type == "standard":
        DO = Dropout
    else:
        raise ValueError(
            f"dropout_type must be one of ['spatial', 'standard'], got {dropout_type}"
        )

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = DO(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def build_vanilla_unet(input_shape,
                dropout=0.5,
                filters=64,
                num_layers=4,
                learning_rate=0.001,
                ssim=False):  # 'sigmoid' or 'softmax'
    # Build U-Net model
    print('Building UNet model')
    # with mirrored_strategy.scope():
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')
        down_layers.append(x)
        x = MaxPooling2D((2, 2), strides=2)(x)
        filters = filters * 2  # double the number of filters with each layer

    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='valid')(x)

        ch, cw = get_crop_shape(int_shape(conv), int_shape(x))
        conv = Cropping2D(cropping=(ch, cw))(conv)

        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=False, dropout=0.0, padding='valid')

    # outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    # optimizer = keras.optimizers.Adam(learning_rate=0.001)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    if ssim:
        model.compile(optimizer=optimizer, loss=ssim_loss, metrics=[ssim_loss, 'mse'])
    else:
        model.compile(optimizer=optimizer, loss='mse')
    return model


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


def build_model(input_shape, f_1, f_2, k_1, k_2, s_1, s_2, d, learning_rate, ssim=True):
    """
    Build Keras model with different hyperparameters

    Note that adding dense_dim substantially increases network size.
    """
    print('Building basic model')
    input_h, input_w = input_shape
    #input_h, input_w = 48, 48
    # perform convolution on input images twice
    #inputs = Input(shape=(48, 48, 2))
    inputs = Input(shape=(input_h, input_w, 2))
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
        #pixel_vectors.append(Dense(576, activation='sigmoid')(h))
        pixel_vectors.append(Dense(int((input_h/2)**2), activation='sigmoid')(h))
    outputs = []
    for i in range(4):
        #outputs.append(Reshape((24, 24))(pixel_vectors[i]))
        outputs.append(Reshape(((int(input_h/2), int(input_w/2))))(pixel_vectors[i]))
    # concatenate predicted segments together
    top = Concatenate(axis=1)([outputs[0], outputs[1]])
    bottom = Concatenate(axis=1)([outputs[2], outputs[3]])
    final = Concatenate(axis=2)([top, bottom])
    final = tf.expand_dims(final, axis=-1)
    model = Model(inputs, final)
    # add loss function
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    if ssim:
        model.compile(optimizer=optimizer, loss=ssim_loss, metrics=[ssim_loss, 'mse'])
    else:
        model.compile(optimizer=optimizer, loss='mse')

    return model

    
def train(model, train_generator, valid_generator, checkpoint_path, model_folder, epochs, ssim=False):
    if ssim:
        loss_monitor = 'val_ssim_loss'
        #loss_monitor = 'ssim_loss'
    else:
        loss_monitor = 'val_mse'
        #loss_monitor = 'mse'
    #edited some things here 'ssim_loss' -> validation loss
    my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=f'{checkpoint_path}/best_checkpoint.h5', monitor=loss_monitor, mode='min', save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor=loss_monitor, factor=0.5, patience=5, min_lr=0.00001),
    keras.callbacks.EarlyStopping(monitor=loss_monitor, mode='min', patience=10)
    ]
    
    #model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val))
    #added validation maybe it won't work
    # print('x_train', x_train.shape, 'y_train', y_train.shape, 'x_val', x_val.shape, 'y_val', y_val.shape)
    history = model.fit(x=train_generator, validation_data=valid_generator, epochs=epochs, batch_size=batch_size, callbacks=my_callbacks)
    #model.save(model_name)
    
    for key in history.history.keys():
        plt.figure(figsize=(10, 10))
        plt.plot(history.history[key], label='%s'%key)
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'{model_folder}plots/{key}.png')
        plt.close()


if __name__=="__main__":
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
    f_1 = model_params['F_1']
    f_2 = model_params['F_2']
    k_1 = model_params['K_1']
    k_2 = model_params['K_2']
    s_1 = model_params['S_1']
    s_2 = model_params['S_2']
    d = model_params['D']

    batch_size = 1

    # model_type = 'unet'
    model_type = 'basic'
    ssim = True

    input_shape = (512, 512)

    if model_type == 'unet':
        output_shape = (324, 324)
    elif model_type == 'basic':
        output_shape = (512, 512)
    else:
        raise NotImplementedError
 
    # global_batch_size = batch_size * mirrored_strategy.num_replicas_in_sync
    # batch_size = global_batch_size
    
    # train_file = sys.argv[2]

    # model_name = train_file[:-5]+"_model.h5"

    h_params_file = sys.argv[2]
    
    with open (h_params_file) as f:
        h_params = json.load(f)
    #h_params = json.load(h_params_file)

    TILT_SERIES_DIR = h_params['TILT_SERIES_DIR']
    # FRAME_GAPS = h_params['FRAME_GAPS']
    FRAME_GAPS = [2] 
    NOISE_LEVELS = h_params['NOISE_LEVELS']
    NOISE_LEVELS = [] 
    REP = h_params['REP']
    TRAIN_TEST_SPLIT = h_params['TRAIN_TEST_SPLIT']
    epochs = 100 
    print('Loading Data...')
    with h5py.File('processed_ts_data_sample.h5', 'r') as hf:
        ts_data = hf['ts_data'][:]
    #ts_data = ts_data[:20]

    print('ts_data', ts_data.shape, type(ts_data))


    print('Creating generators...')

    # for ts_data in tilt_series_list[:1]:
    # idx = np.random.choice(len(ts_data), len(ts_data))
    # train_idx = idx[:int(TRAIN_TEST_SPLIT * len(idx))]
    # valid_idx = idx[int(TRAIN_TEST_SPLIT * len(idx)):]

    train_data = ts_data[:int(TRAIN_TEST_SPLIT * len(ts_data))] 
    valid_data = ts_data[int(TRAIN_TEST_SPLIT * len(ts_data)):] 
    print(len(train_data), len(valid_data))

    # train_data = train_data[:256]
    # valid_data = valid_data[:128]
    # for delta in FRAME_GAPS[:1]:
    # configs = {'model': model_params, 'data': data_params 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format as YearMonthDay_HourMinuteSecond

    learning_rate = 0.01
    configs = h_params.copy()
    configs.update(model_params)
    configs['model_type'] = model_type
    configs['input_shape'] = input_shape
    configs['output_shape'] = output_shape
    configs['ssim'] = ssim
    configs['data_len'] = len(ts_data)
    configs['train_len'] = len(train_data)
    configs['valid_len'] = len(valid_data)
    configs['learning_rate'] = learning_rate

    for delta in FRAME_GAPS:
        if len(NOISE_LEVELS) == 0:
            model_name = f"frame_gap_{delta}_model_{timestamp}"
        else:
            model_name = f"frame_gap_{delta}_aberated_model_{timestamp}"

        model_folder = f'./trained_models/{model_name}/'
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        if not os.path.exists(model_folder + 'plots/'):
            os.mkdir(model_folder + 'plots/')

        checkpoint_path = f'./checkpoints/{model_name}/'
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        configs['model_name'] = model_name
        configs['delta'] = delta
        configs_file = f'./{model_folder}/config.json'
        with open (configs_file, 'w') as f:
            json.dump(configs, f)
        

        train_generator = DataGenerator(train_data, delta, dim=input_shape, dim_out=output_shape, noise_levels=NOISE_LEVELS, batch_size=batch_size, n_channels=2, shuffle=True)
        valid_generator = DataGenerator(valid_data, delta, dim=input_shape, dim_out=output_shape, noise_levels=[], batch_size=batch_size, n_channels=2, shuffle=False)
        print(model_folder)
        print('Gap:', delta)
        if model_type == 'basic':
            model = build_model(input_shape, f_1, f_2, k_1, k_2, s_1, s_2, d, learning_rate=learning_rate, ssim=ssim)
        elif model_type == 'unet':
            model = build_vanilla_unet((input_shape[0], input_shape[1], 2), learning_rate=learning_rate, ssim=ssim)
        else:
            raise NotImplementedError
        train(model, train_generator, valid_generator, checkpoint_path=checkpoint_path, model_folder=model_folder, epochs=epochs, ssim=ssim)

    print('Done')

    # for x, y in train_generator:
    #     print(x.shape, y.shape)

    # f  = h5py.File(train_file)
    # x_train = f['x_train']
    # x_train = np.array(x_train)
    # #x_train = x_train[:, :48, :48, :]
    # shape_train = x_train.shape[0]
    # shape_train = int(shape_train/float(batch_size)) * batch_size # ssim needs to have the same batch size for all the mini batches
    # x_train = x_train[:shape_train,:,:,:]
    # y_train = f['y_train']
    # y_train = np.array(y_train)
    # #y_train = np.
    # y_train = y_train[:, :324, :324]
    # y_train = y_train[:shape_train,:,:]
    # x_val = f['x_val']
    # x_val = np.array(x_val)
    # #x_val = x_val[:, :48, :48, :]
    # y_val = f['y_val']
    # y_val = np.array(y_val)
    # y_val = y_val[:, :324, :324]

    
    #model = build_model(f_1, f_2, k_1, k_2, s_1, s_2, d, ssim=False)
    # model = build_model(f_1, f_2, k_1, k_2, s_1, s_2, d, ssim=True)

