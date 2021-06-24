import numpy as np
import keras
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, image_dir, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.image_dir = image_dir
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
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

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 512, 512, 2))
        y = np.empty((self.batch_size, 512, 512, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image = np.load(os.path.join(self.image_dir, "{}.npy".format(ID)))
    
            # Store sample
            X[i, :, :, 0] = image[:, :, 0]
            X[i, :, :, 1] = image[:, :, 2]

            # Store class
            y[i, :, :, 0] = image[:, :, 1]

        return X, y


class ResidualDataGenerator(DataGenerator):
    
    def __init__(self, **kwargs):
        super(ResidualDataGenerator, self).__init__(**kwargs)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 512, 512, 2))
        y = np.empty((self.batch_size, 512, 512, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image = np.load(os.path.join(self.image_dir, "{}.npy".format(ID)))
    
            # Store sample
            X[i, :, :, 0] = image[:, :, 0]
            X[i, :, :, 1] = image[:, :, 2]
            
            mean = (image[:, :, 0] + image[:, : 2])/2.
            # Store class
            y[i, :, :, 0] = image[:, :, 1] - mean

        return X, y

class Res2ResDataGenerator(DataGenerator):
    
    def __init__(self, **kwargs):
        super(ResidualDataGenerator, self).__init__(**kwargs)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 512, 512, 2))
        y = np.empty((self.batch_size, 512, 512, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image = np.load(os.path.join(self.image_dir, "{}.npy".format(ID)))
    
            mean = (image[:, :, 0] + image[:, : 2])/2.
            
            # Store sample
            X[i, :, :, 0] = image[:, :, 0] - mean
            X[i, :, :, 1] = image[:, :, 2] - mean
            
            # Store class
            y[i, :, :, 0] = image[:, :, 1] - mean

        return X, y


