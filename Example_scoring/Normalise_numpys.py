import pathlib
import numpy as np
import sys

def __main__():
    """
    Takes numpy stack (NxNxM) and:
    - Normalises it between [0,1] wrt to each slice 
    - converts data to float 32
    - saves the numpy stack (overwrites the existing numpy file)
    """
    
    
    
    numpy_path = sys.argv[1]
    files = sorted(numpy_path.glob('**/**/*npy')) # this is important to check otherwise it will run without error 
    print(len(files))
   
    def normalise_slice(s):
        """
        takes an n x m slice and normalises it between [0,1]
        """

        s = s - s.min()
        s = s / s.max()
        return s

    def normalise_stack(stack):
        """
        takes an  n x n x M stack and normalises each slice wrt to itself. 

        """

        for index in range(stack.shape[2]):
            stack[:,:,index] = normalise_slice(stack[:,:,index])

        return stack



    for file in files:
        stack = np.load(file)
        stack = normalise_stack(stack)
        stack = stack.astype('float32')

        np.save(file, stack)

        
if __name__ == __main__:
    __main__()





