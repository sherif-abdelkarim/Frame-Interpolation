import pathlib
import numpy as np
import pathlib
import sys
import os
import glob
import h5py


def __main__():
    
    #set the emd_path from sys argument
    emd_path = pathlib.Path(sys.argv[1])
    
    #these are all the steps I wanted to generate things 
    step_sizes = [1, 2, 4, 12, 20] #[i for i in range(2, 21, 2)]
    #set a counter _ not important just to check
    count=0
    
    for folder in sorted(emd_path.glob('*/')):
        #look through all 101 crystals 
        
        #get the emd file for the crystal I know there is only one emd file in there. 
        file = sorted(folder.glob('*.emd'))[0]
        
        #open the file
        stack = h5py.File(file, 'r')
        
        #get the data 
        data = stack['data/tomography/data'][:,:,:]
        
        #close file
        stack.close()

        for step in step_sizes:
            
            #generate the output file name
            out_name = file.name[:-4] + f'_step_{step}.npy'
            
            #generate the output path
            out_path = folder / f'step_{step}' / out_name

            #work out what slices are needed
            step_slices = [i for i in np.arange(0, 181, step)]

            #generate the new step sized data
            new_stack = data[:,:, step_slices]

            #I fucked up and will have to run the NPY-EMD Script again. 
            np.save(out_path, new_stack)

            count += 1
    print(count)
    
    
if __name__ == '__main__':
    __main__()


