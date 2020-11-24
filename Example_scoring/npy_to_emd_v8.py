# dependencies
import h5py 
import glob
import numpy as np
import datetime as dt 
import sys
from pathlib import Path


def batch_convert_npy_to_emd():
    """

    convert all .npy files from one folder to .emd in another folder
    """
    
   
    in_path = sys.argv[1]
    in_path = Path(in_path)

    out_path = sys.argv[2]
    out_path = Path(out_path)

    #glob all the files
    files = in_path.glob('*.npy')

    

    #for loop to genreate emd file  
    for file in files:



        #get current file name
        name = file.name

        #strip the .npy and add the .emd extension
        name = name.split('.')[0] + '.emd'

        #set up the path/name.emd to create 
        filename = out_path / name

        #load the data
        file= np.load(file)

        #create the h5py file in writting mode
        f = h5py.File(filename, 'w')



        #set attributes
        f.attrs['version_minor'] = np.array([2], dtype=np.uint32)
        f.attrs['version_major'] = np.array([0], dtype=np.uint32)

        #set up the file structure 
        grp_exp = f.create_group('data')
        #This has no attributes

        #create tomography group
        grp_tom = grp_exp.create_group('tomography')
        #set tomography attributes
        grp_tom.attrs[b'EMD_group_type'] = np.array([1], dtype=np.uint32)

        #create data placeholder
        data = grp_tom.create_dataset('data',
                                    (file.shape[0], file.shape[1], file.shape[2]),
                                    dtype=np.float)
        #set data attributes
        data.attrs['name'] = np.array([b'ImageScalars'], dtype="|S")

        #create dim 1 placeholder
        dim1 = grp_tom.create_dataset('dim1', (file.shape[0], 1), dtype=np.uint32)
        #set dim1 attributes
        dim1.attrs['name'] = np.array([b'x'], dtype="|S")
        dim1.attrs['units'] = np.array([b'[n_m]'], dtype="|S")

        #create dim 2 placeholder
        dim2 = grp_tom.create_dataset('dim2', (file.shape[1], 1), dtype=np.uint32 )
        #set dim2 attributes
        dim2.attrs['name'] = np.array([b'y'], dtype="|S")
        dim2.attrs['units'] = np.array([b'[n_m]'], dtype="|S")

        #create dim 3 placeholder
        dim3 = grp_tom.create_dataset('dim3', (file.shape[2], 1), dtype=np.uint32)
        #set dim3 attributes
        dim3.attrs['name'] = np.array([b'z'], dtype="|S")
        dim3.attrs['units'] = np.array([b'[n_m]'], dtype='|S')

        #create tomviz_scalars group
        tomviz_scalars = grp_tom.create_group('tomviz_scalars', [])
        #link image_scalars to data
        tomviz_scalars['ImageScalars'] = data
        image_scalars = tomviz_scalars['ImageScalars']

        #set image_scalars attribute
        image_scalars.attrs['name'] = np.array([b'ImageScalars'], dtype="|S")
        #add the data to each palceholder
        data[:, :, :] = file.astype(np.float64)
        dim1[:] = np.arange(0, file.shape[0], dtype=np.float32).reshape((file.shape[0], 1))
        dim2[:] = np.arange(0, file.shape[1], dtype=np.float32).reshape((file.shape[1], 1))
        dim3[:] = np.arange(0, file.shape[2], dtype=np.float32).reshape((file.shape[2], 1))
        
        #close the file
        f.close()

def __main__():
    batch_convert_npy_to_emd()


if __name__ == "__main__":
    __main__()
