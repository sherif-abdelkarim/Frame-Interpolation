#!/bin/bash
import pathlib
import sys
import os
import subprocess

def state_key(string):
    
    string = int(string.name.split('_')[-1][:-5])
    
    return string


def __main__():
    
    def state_key(string):
    
        string = int(string.name.split('_')[-1][:-5])
    
        return string  
    
    base_path = pathlib.Path(sys.argv[1])
    recon_state_path = base_path / 'Tomviz_files'
    emd_path = base_path / 'emds_interpolated'
    
    #load all the state files and sorted im by step size
    state_files = sorted(recon_state_path.glob('*tvsm'), key=state_key)
    #load all the crystal folders
    folders  = sorted(emd_path.glob('*/'))
    print(len(folders))
    #search over all crystla folders 
    for folder in folders:
        #search all step size folders from 1 --> 20
        for i in [2, 4, 12, 20]: #range(2, 21, 2):
            #pick the specific step folder
            fold = folder / f'step_{i}'
            #load the emd file
            file = sorted(fold.glob('*ml.emd'))[0] #do for ml int file
            
            #set the name of the reconstructed emd file
            out_name = file.name[:-4] + '_recon.emd'
            #set the out_put path
            out_path = file.parent / out_name
            
            state = state_files[(i-2)//2]
            state = state.absolute().as_posix()
            state = state.replace(' ', '\\ ')
            state = state.replace('(', '\\(')
            state = state.replace(')', '\\)')
            
            out_path = out_path.absolute().as_posix()
            out_path = out_path.replace(' ', '\\ ')
            out_path = out_path.replace('(', '\\(')
            out_path = out_path.replace(')', '\\)')
            
            file = file.absolute().as_posix()
            file = file.replace(' ', '\\ ')
            file = file.replace('(', '\\(')
            file = file.replace(')', '\\)')
            
            
            #print(f'{state}  |||  {file}  |||  {out_path}')
            
            os.system(f'tomviz-pipeline -s {state} -d {file} -o {out_path}')
            #subprocess.call(['/bin/bash', f'tomviz-pipeline -s {state} -d {file}'])
            
    for folder in folders:
        #search all step size folders from 2 --> 20
        for i in [2, 4, 12, 20]: #range(2, 21,2):
            #pick the specific step folder
            fold = folder / f'step_{i}'
            #load the emd file
            file = sorted(fold.glob('*lin.emd'))[0] #do for linear interpolated file
            
            #set the name of the reconstructed emd file
            out_name = file.name[:-4] + '_recon.emd'
            #set the out_put path
            out_path = file.parent / out_name
            #state files have tilts for 1->20 with index of 0->19
            #state for interpolations need to derived from the step folder size 2, 4, 6, 10, 12, 18, 20
            
            state = state_files[(i-2)//2]
            state = state.absolute().as_posix()
            state = state.replace(' ', '\\ ')
            state = state.replace('(', '\\(')
            state = state.replace(')', '\\)')
            
            out_path = out_path.absolute().as_posix()
            out_path = out_path.replace(' ', '\\ ')
            out_path = out_path.replace('(', '\\(')
            out_path = out_path.replace(')', '\\)')
            
            file = file.absolute().as_posix()
            file = file.replace(' ', '\\ ')
            file = file.replace('(', '\\(')
            file = file.replace(')', '\\)')
            
            
            print(f'{state}  |||  {file}  |||  {out_path}')
            
            os.system(f'tomviz-pipeline -s {state} -d {file} -o {out_path}')
            #subprocess.call(['/bin/bash', f'tomviz-pipeline -s {state} -d {file}'])

    for folder in folders:
        #search all step size folders from 2 --> 20
        for i in [1]: #range(2, 21,2):
            #pick the specific step folder
            fold = folder / f'step_{i}'
            #load the emd file
            file = sorted(fold.glob('*step_1.emd'))[0] #do for linear interpolated file
            
            #set the name of the reconstructed emd file
            out_name = file.name[:-4] + '_orig_recon.emd'
            #set the out_put path
            out_path = file.parent / out_name
            
            state = state_files[(i-1)//2]
            state = state.absolute().as_posix()
            state = state.replace(' ', '\\ ')
            state = state.replace('(', '\\(')
            state = state.replace(')', '\\)')
            
            out_path = out_path.absolute().as_posix()
            out_path = out_path.replace(' ', '\\ ')
            out_path = out_path.replace('(', '\\(')
            out_path = out_path.replace(')', '\\)')
            
            file = file.absolute().as_posix()
            file = file.replace(' ', '\\ ')
            file = file.replace('(', '\\(')
            file = file.replace(')', '\\)')
            
            
            print(f'{state}  |||  {file}  |||  {out_path}')
            
            os.system(f'tomviz-pipeline -s {state} -d {file} -o {out_path}')
            #subprocess.call(['/bin/bash', f'tomviz-pipeline -s {state} -d {file}'])
            
            

if __name__ == '__main__':
    __main__()
