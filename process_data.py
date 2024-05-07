import sys
import json
import os

import numpy as np
import h5py

if __name__=="__main__":
    sample = True
    h_params_file = sys.argv[1]

    with open (h_params_file) as f:
        h_params = json.load(f)

    TILT_SERIES_DIR = h_params['TILT_SERIES_DIR']
    FRAME_GAPS = h_params['FRAME_GAPS']
    NOISE_LEVELS = h_params['NOISE_LEVELS']
    REP = h_params['REP']
    TRAIN_TEST_SPLIT = h_params['TRAIN_TEST_SPLIT']

    tilt_series_list = []
    tilt_series_dir = os.listdir(TILT_SERIES_DIR)
    print('Loading tilt series...')
    for f in tilt_series_dir:
        if f.split(".")[1] == "npy":
            # convert to float32 to save space
            tilt_series_list.append(np.float32(np.load(os.path.join(TILT_SERIES_DIR, f))))
    print('Done')
    # make sure that the values are exactly between 0 and 1
    tilt_series_list = [x / np.max(x) for x in tilt_series_list]
    tilt_series_list = [np.swapaxes(x, 0, 2)[:, :, :] for x in tilt_series_list]
    # print(len(tilt_series_list))
    # print([t.shape for t in tilt_series_list])
    ts_data = np.concatenate(tilt_series_list, axis=0)
    filename = 'processed_ts_data'
    if sample:
        ts_data = ts_data[:5000]
        filename += '_sample'
    print('ts_data', ts_data.shape, type(ts_data))
    
    # Saving as HDF5
    with h5py.File(f'{filename}.h5', 'w') as hf:
        hf.create_dataset('ts_data', data=ts_data, compression='gzip')  # You can specify the compression level with compression_opts
    print('Data Saved!')
