import glob
import pathlib
import os 
import sys



def __main__():
    path = pathlib.Path(sys.argv[1])
    dirs = sorted(path.glob('**/'))
    
    for crystal in range(1, len(dirs)):
        for tilt in [1, 2, 4, 12, 20]:
            os.mkdir(dirs[crystal] / f'step_{tilt}')




if __name__ == "__main__":
    __main__()
