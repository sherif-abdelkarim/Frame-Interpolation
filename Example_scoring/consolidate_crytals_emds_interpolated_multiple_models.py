import itk
import pathlib
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import skimage
from skimage.filters import threshold_otsu
from scipy  import signal 
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
import pandas as pd
import sys


def step_folder_key(string):
    '''
    String sorter for step folders.
    Which have the format: 
        step_n 
    '''

    string = int(string.name.split('_')[-1])

    return string


def set_paths(base_path, emd_folder_name):
    """
    Inputs:
    base_path: string to the path required. This will be sys.argv[1]
    emd_folder_name: name the emd folder 
    
    """
    
    base_path = pathlib.Path(base_path)
    emds_path = base_path / emd_folder_name
    
    return base_path, emds_path


def get_paths_dictionary(step_folders, glob_str):
    '''
    ***
    Little janky - requires glob_str to return list len 1 
    ***
    
    Inputs:
    
    step_folders: list of paths to all the step folders in 
    a crystal folder
    glob_str: string to glob on. 
    
    Outputs: 
    paths_dic: dictionary with key: step_n, item: recon_file_path
    
    '''
    
    #create an empty dictionary 
    paths_dic = {}

    #loop over all step folders, and get the reconstruction
    #file and append to dictionary

    for step_folder in step_folders:
        file = sorted(step_folder.glob(glob_str))[0]
        print(step_folder.name)
        paths_dic[f'{step_folder.name}'] = file 
        
    return paths_dic

def get_paths_dictionary_interpolated_gt(step_folder, glob_str):
    '''
    ***
    Little janky - requires glob_str to return list len 1 
    ***
    
    Inputs:
    
    step_folders: list of paths to all the step folders in 
    a crystal folder
    glob_str: string to glob on. 
    
    Outputs: 
    paths_dic: dictionary with key: step_n, item: recon_file_path
    
    '''
    
    #create an empty dictionary 
    paths_dic = {}

    #loop over all step folders, and get the reconstruction
    #file and append to dictionary

    
    file = sorted(step_folder.glob(glob_str))[0]
    paths_dic[f'{step_folder.name}'] = file 

    return paths_dic

def get_data_dictionary(paths_dic):
    """
    ***
    Only works on the emd written by tomviz
    ***
        
    Inputs:
    paths_dic: dictionary with key: step_n, item: recon_file_path
    
    
    outputs: 
    
    data_dic: dictionary with key:step_n, item: recon_file_data(nxnxn array) 
    
    """
    
    #create data_dic
    data_dic = {}
    
    #loop over all keys(step_sizes) in the paths dic, load file
    #append data to data_dic
    for key in paths_dic:
        with h5py.File(paths_dic[f'{key}'], 'r') as recon: #open file 
            data = recon['data/tomography/data'][:,:,:] #get the data
        data_dic[key] = data
    
    return data_dic

    
    
def otsu_threshold_process(data_dic):
    """
    Inputs:
    
    data_dic: dictionary with key: step_n, item: recon_file_data(nxnxn array) array
    
    Outputs:
    
    otsu_dic: dictionary with key: step_n, item:binary otsu threshold mask
    
    masked_data_dic: dictionary with key: step_n, item:masked data (otsu mask * data)

    """
    
    #create a dictionary for binary otsu image stacks
    otsu_dic = {}
    #create a dicionary for the masked data
    masked_data_dic = {}

    #loop over all the image stacks 
    for key in data_dic:
        
        #get_data stack
        data = data_dic[f'{key}']  
          
        #convert the image to itk style
        itk_image = itk.image_from_array(data_dic[f'{key}'])

        #get the itk type
        itk_input_image_type = type(itk_image)

        #set the new itk image type
        itk_threshold_image_type = itk_input_image_type

        #create instance of otsu filter
        otsu_filter = itk.OtsuMultipleThresholdsImageFilter[itk_input_image_type, itk_threshold_image_type].New()
        #i want a binary threshold so 1
        otsu_filter.SetNumberOfThresholds(1)
        #not sure what this is but don't think it matters for our case
        otsu_filter.SetValleyEmphasis(False)
        #set the input image as itk_image
        otsu_filter.SetInput(itk_image)
        #run the filter algo
        otsu_filter.Update()

        #get binary mask
        itk_mask = otsu_filter.GetOutput() 

        data_mask = itk.array_from_image(itk_mask)

        #save it as an array
        otsu_mask = itk.array_from_image(itk_mask)
        #add to dic
        otsu_dic[f'{key}'] = otsu_mask

        #returns the otsu mask but with orginal array values 
        masked_data = data * data_mask
        #add to dic
        masked_data_dic[f'{key}'] = masked_data
        
    return otsu_dic, masked_data_dic


def score_and_make_dataframe_noise_interpolation(otsu_dic, masked_data_dic,
                                                 gt_otsu_dic, gt_masked_data_dic,  
                                                  folder, interpolation):
    """
    Inputs:
    
    otsu_dic: dictionary with key: step_n, item:binary otsu threshold mask
    masked_data_dic: dictionary with key: step_n, item:masked data (otsu mask * data)
    folder: pathlib to directory 
    
    Outputs:
    
    Saves a pkl file in the parent folder
       
    """
    
    #just to keep a check everything is in order 
    step_list = []

    #self similarity index
    masked_ssim = []
    ssim_otsu = []
    
    #normalised mean square error
    masked_nrmse = []
    nrmse_otsu = []

    masked_mse = []
    mse_otsu = []

    ####
    #THESE AREN"T CURRENTLY USED
    ####
      
    
    #cross correlation
    masked_xc = []
    xc_otsu = []

    #mean values
    masked_mean = []
    mean_otsu = []

    #varience in the stack
    masked_var = []
    var_otsu = []


    # Loop over all crystal step sizes
    for key in otsu_dic:
        
        #check things come out in order
        step_list.append(key.split('_')[-1])

        #using step 1, as ground truth
        ssim_otsu.append(compare_ssim(gt_otsu_dic[f'step_1'], otsu_dic[f'{key}']))
        nrmse_otsu.append(compare_nrmse(gt_otsu_dic[f'step_1'], otsu_dic[f'{key}']))
        mse_otsu.append(compare_mse(gt_otsu_dic[f'step_1'], otsu_dic[f'{key}']))
        xc_otsu.append(signal.correlate(gt_otsu_dic[f'step_1'], otsu_dic[f'{key}']).sum())
        mean_otsu.append(otsu_dic[f'{key}'].mean())
        var_otsu.append(otsu_dic[f'{key}'].var())

        
        masked_ssim.append(compare_ssim(gt_masked_data_dic[f'step_1'], masked_data_dic[f'{key}']))
        masked_nrmse.append(compare_nrmse(gt_masked_data_dic[f'step_1'], masked_data_dic[f'{key}']))
        masked_mse.append(compare_mse(gt_masked_data_dic[f'step_1'], masked_data_dic[f'{key}']))
        masked_xc.append(signal.correlate(gt_masked_data_dic[f'step_1'], masked_data_dic[f'{key}']).sum())
        masked_mean.append(masked_data_dic[f'{key}'].mean())
        masked_var.append(masked_data_dic[f'{key}'].var())

    #add all this stuff to a dataframe
    df = pd.DataFrame(
            {'step_size' : step_list,
             'ssim_otsu' : ssim_otsu,
             'nrmse_otsu' : nrmse_otsu,
             'mse_otsu' : mse_otsu,
             'xc_otsu' : xc_otsu,
             'mean_otsu' : mean_otsu,
             'var_otsu' : var_otsu,
             'masked_ssim' : masked_ssim,
             'masked_nrmse' : masked_nrmse,
             'masked_mse' : masked_mse,
             'masked_xc' : masked_xc,
             'masked_mean' : masked_mean,
             'masked_var' : masked_var         
                }

            )

    #normalised a couple of things with respect to values of step 1
    df['norm_xc_otsu'] = df.apply(lambda row: row.xc_otsu / df.xc_otsu.iloc[0], axis=1)
    df['norm_masked_xc'] = df.apply(lambda row: row.masked_xc / df.masked_xc.iloc[0], axis=1)

    #create name of hte output file
    df_name = folder.name + f'interpolation_{interpolation}.pkl'
    #create the full output path
    out_path = folder / df_name
    #save the data frame to the crystal folder
    pd.to_pickle(df, out_path)
    
    
    return out_path


def score_and_make_dataframe(otsu_dic, masked_data_dic, folder):
    """
    Inputs:
    
    otsu_dic: dictionary with key: step_n, item:binary otsu threshold mask
    masked_data_dic: dictionary with key: step_n, item:masked data (otsu mask * data)
    folder: pathlib to directory 
    
    Outputs:
    
    Saves a pkl file in the parent folder
       
    """
    
    #just to keep a check everything is in order 
    step_list = []

    #self similarity index
    masked_ssim = []
    ssim_otsu = []
    
    #MSE
    masked_mse = []
    mse_otsu = []
    
    ####
    #THESE AREN"T CURRENTLY USED
    ####
    
    #normalised mean square error
    masked_nrmse = []
    nrmse_otsu = []

    #cross correlation
    masked_xc = []
    xc_otsu = []

    #mean values
    masked_mean = []
    mean_otsu = []

    #varience in the stack
    masked_var = []
    var_otsu = []


    # Loop over all crystal step sizes
    for key in otsu_dic:
        
        #check things come out in order
        step_list.append(key.split('_')[-1])

        #using step 1, as ground truth
        ssim_otsu.append(compare_ssim(gt_otsu_dic[f'step_1'], otsu_dic[f'{key}']))
        nrmse_otsu.append(compare_nrmse(gt_otsu_dic[f'step_1'], otsu_dic[f'{key}']))
        mse_otsu.append(compare_mse(gt_otsu_dic[f'step_1'], otsu_dic[f'{key}']))
        xc_otsu.append(signal.correlate(gt_otsu_dic[f'step_1'], otsu_dic[f'{key}']).sum())
        mean_otsu.append(otsu_dic[f'{key}'].mean())
        var_otsu.append(otsu_dic[f'{key}'].var())

        
        masked_ssim.append(compare_ssim(gt_masked_data_dic[f'step_1'], masked_data_dic[f'{key}']))
        masked_nrmse.append(compare_nrmse(gt_masked_data_dic[f'step_1'], masked_data_dic[f'{key}']))
        masked_mse.append(compare_mse(gt_masked_data_dic[f'step_1'], masked_data_dic[f'{key}']))
        masked_xc.append(signal.correlate(gt_masked_data_dic[f'step_1'], masked_data_dic[f'{key}']).sum())
        masked_mean.append(masked_data_dic[f'{key}'].mean())
        masked_var.append(masked_data_dic[f'{key}'].var())

    #add all this stuff to a dataframe
    df = pd.DataFrame(
            {'step_size' : step_list,
             'ssim_otsu' : ssim_otsu,
             'nrmse_otsu' : nrmse_otsu,
             'mse_otsu' : mse_otsu,
             'xc_otsu' : xc_otsu,
             'mean_otsu' : mean_otsu,
             'var_otsu' : var_otsu,
             'masked_ssim' : masked_ssim,
             'masked_nrmse' : masked_nrmse,
             'masked_mse' : masked_mse,
             'masked_xc' : masked_xc,
             'masked_mean' : masked_mean,
             'masked_var' : masked_var         
                }

            )

    #normalised a couple of things with respect to values of step 1
    df['norm_xc_otsu'] = df.apply(lambda row: row.xc_otsu / df.xc_otsu.iloc[0], axis=1)
    df['norm_masked_xc'] = df.apply(lambda row: row.masked_xc / df.masked_xc.iloc[0], axis=1)

    #create name of hte output file
    df_name = folder.name + '.pkl'
    #create the full output path
    out_path = folder / df_name
    #save the data frame to the crystal folder
    pd.to_pickle(df, out_path)
    
    return out_path
    
    
def __main__():
    """

    
    """
    #set the name of the emd folder
    emd_folder_name = 'emds_interpolated'
    
    #set paths (basd_path is redundant)
    #base_path, emds_path = set_paths(sys.argv[1], emd_folder_name)
    
    base_path = pathlib.Path(sys.argv[1])
    
    emds_path = base_path / emd_folder_name
    #get all the crystal folders
    folders = sorted(emds_path.glob('*/'))
    
    interpolations = ['lin', 'ml']

    #loop over the folders parent folder of the each crystal:
    for folder in folders:
                
        #get all the step folders i.e. step_1 --> step_20
        step_folders = sorted(folder.glob('step*/'), key=step_folder_key)
        
        #get the ground truth file path i.e. step_1
        ground_truth_path_dic = get_paths_dictionary_interpolated_gt(step_folders[0], '*orig_recon.emd')
        
        #check the name of ground truth file
        print(ground_truth_path_dic['step_1'].name)
        
        #get the ground truth data
        ground_truth_data_dic = get_data_dictionary(ground_truth_path_dic)
        
        # get the thesholds data
        ground_truth_otsu_mask_dic, ground_truth_masked_data_dic = otsu_threshold_process(ground_truth_data_dic)
        
        
        
        #loop over the interpolation types i.e. lin and ml
        for interpolation in interpolations:

            interpolation_str = '*' + interpolation + '_recon.emd'
            
            #print the name of the interpolation type
            print(interpolation_str)
            
            #print the name of the folder its working on
            print(f'processing: {folder.name}')
            
                    
            #get paths for all the recon files in step folders
            paths_dic = get_paths_dictionary(step_folders[1:], interpolation_str)
            
            #get the data of for each reconstruction
            data_dic = get_data_dictionary(paths_dic)
            
            #process the reconstructions by otsu threshold
            otsu_dic, masked_data_dic = otsu_threshold_process(data_dic)
            
            #score the reconstructions wrt to step_1 and save dataframe
            score_and_make_dataframe_noise_interpolation(otsu_dic, masked_data_dic,
                                                         ground_truth_otsu_mask_dic, ground_truth_masked_data_dic,
                                                          folder, interpolation)
                    

    
        
        
        print(f'finsined processing: {folder.name}')
        
        
    print('All done')
    
    
if __name__ ==  '__main__':
    __main__()
    
    
        
            
