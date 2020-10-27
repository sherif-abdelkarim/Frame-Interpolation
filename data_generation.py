import numpy as np
import os
import h5py
import random

def setup_frames(ts_data, delta):
    """
    ts_data: (list) a list of tilt series
    n_series: (int) number of tilt series in the stack
    n_frames: (int) number of frames in each tilt series
    delta: (int) number of frames to skip

    For example, a data set with 1010 frames can have 10 tilt series with
    101 frames for each series (n_series = 10, n_frames = 101).
    """
    n_series = len(ts_data)
    n_frames, w, h = ts_data[0].shape
    x_train = []
    y_train = []

    for tilt_serie in ts_data:
        for index in range(delta, n_frames, 1):
            #target_index = delta/2
            #assert type(target_index)==np.int
            first_frame_index = index-delta
            second_frame_index = index
            target_index = int((second_frame_index + first_frame_index)/2.)
            print(target_index)
            stacked = np.zeros((w, h, 2))
            stacked[:, :, 0] = tilt_serie[first_frame_index, :, :]
            stacked[:, :, 1] = tilt_serie[second_frame_index, :, :]
            target = tilt_serie[target_index, :, :] 
            x_train.append(stacked)
            y_train.append(target)

    x_train = np.stack(x_train)
    y_train = np.stack(y_train)

    return x_train, y_train

def poisson_noise(data, scale=500.0):
    """
    Add Poisson noise by scaling the pixel values (between 0 and 1) to intergers
    then scaling back. The scale is relative and needs to be chosen carefully.

    Note scale must be a float.
    """
    return np.random.poisson(data * scale) / scale


def prepare_data(ts_data, frame_gaps, noise_levels, rep=1, train_test_split=0.9):
    """
    Prepare training and validation data for frame interpolation.
    Note that this method contains many nested loops and may need optimization.

    Args
        frame_gaps: (list) of integers for the number of frames to skip
        noise_levels: (list) of floats for noise scale
        rep: (int) how many repetitions to perform this
    """
    # take out frames from tilt series
    for g in frame_gaps:
        x_list = []
        y_list = []
        x, y = setup_frames(ts_data, g)
        x_list.append(x)
        y_list.append(y)
        # add Poisson noise
        x_noisy_list = []
        for x in x_list:
            for l in noise_levels:
                for j in range(rep):
                    x_noisy_list.append(poisson_noise(x, l))
        # copy target frames
        y_target_list = []
        for y in y_list:
            for l in noise_levels:
                for j in range(rep):
                    #y_target_list.append(poisson_noise(y, l))
                    y_target_list.append(y)

        # shuffle the x and y s
        c = list(zip(x_noisy_list, y_target_list))
        random.shuffle(c)
        x, y = zip(*c)
               
        x = np.concatenate(x)
        y = np.concatenate(y)

        print(x.shape)
        print(y.shape)

        n_data = x.shape[0]

        x_train = x[:int(train_test_split*n_data), :, :]
        x_val = x[int(train_test_split*n_data):, :, :]
        y_train = y[:int(train_test_split*n_data), :, :]
        y_val = y[int(train_test_split*n_data):, :, :]
    
        write_into_h5("frame_gaps_%d_aberated"%g, x_train, y_train, x_val, y_val)
        
        del x_train
        del y_train
        del x_val
        del y_val

def write_into_h5(file_name, x_train, y_train, x_val, y_val):
    f = h5py.File("%s.hdf5"%file_name, 'w')
    #print("writing %s into an hdf5 dataset..."%file_name)
    f.create_dataset("x_train", data=x_train)
    f.create_dataset("y_train", data=y_train)
    f.create_dataset("x_val", data=x_val)
    f.create_dataset("y_val", data=y_val)
    f.close()

if __name__ == "__main__":

    h_params_file = sys.argv[1]
    h_params = json.load(h_params_file)

    TILT_SERIES_DIR = h_params['tilt_series_dir']
    FRAME_GAPS = h_params['frame_gaps']
    NOISE_LEVELS = h_params['noise_levels']
    REP = h_params['rep']
    TRAIN_TEST_SPLIT = h_params['train_test_split']
    

    tilt_series_list = []
    for f in os.listdir(TILT_SERIES_DIR):
        if f.split(".")[1] == "npy":
            # convert to float32 to save space
            tilt_series_list.append(np.float32(np.load(os.path.join(TILT_SERIES_DIR, f))))

    # make sure that the values are exactly between 0 and 1
    tilt_series_list = [x / np.max(x) for x in tilt_series_list]
    # change size from 50 x 50 to 48 x 48
    tilt_series_list = [np.swapaxes(x, 0, 2)[:, 1:49, 1:49] for x in tilt_series_list]
    prepare_data(tilt_series_list, frame_gaps=FRAME_GAPS, noise_levels=NOISE_LEVELS, rep=REP, 
                 train_test_split=TRAIN_TEST_SPLIT)





