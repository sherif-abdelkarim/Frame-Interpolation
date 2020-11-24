import glob
import pathlib
import os 
import sys

def __main__():    
    path = pathlib.Path(sys.argv[1])

    files = path.glob('*emd')

    for f in files:
        os.mkdir(path / f.name[:-4])
    
    

if __name__ == "__main__":
    __main__()
