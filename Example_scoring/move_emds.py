import pathlib
import os
import sys


def __main__():
    path = pathlib.Path(sys.argv[1])
    files = path.glob('*.emd')

    for f in files:
        os.rename(f, path / f.name[:-4] / f.name)
        
if __name__ == '__main__':
    __main__()

