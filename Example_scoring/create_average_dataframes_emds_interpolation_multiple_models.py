import pandas as pd
import numpy as np
import pathlib
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
#import seaborn as sns


def sort_crystals(files):
    #define the crystals 

    classic_elements = ['Pt', 'Au', 'Cu', 'Ag']
    exotic_elements = ['Yb', 'Ir', 'Xe', 'Al']

    classic_sym = ['Octa', 'Iso', 'Deca']
    exotic_sym = ['Fcc']

    classic_types = ['Classic']
    exotic_types = ['CS', 'HC', 'RB', 'SC']

    #create some empty lists
    classic_element_crystals = []
    classic_sym_crystals = []
    classic_type_crystals = []


    # search over the crystals and append to classes there

    #should make this a function 
    for file in files:
        for element in classic_elements:
            if element in file.name: classic_element_crystals.append(file) 
            else: None
    for file in files:
        for sym in classic_sym:                                   
            if sym in file.name: classic_sym_crystals.append(file)
            else: None

    for file in files:        
        for typ in classic_types:
            if typ in file.name: classic_type_crystals.append(file)
            else: None

    exotic_element_crystals = []
    exotic_sym_crystals = []
    exotic_type_crystals = []

    for file in files:
        for element in exotic_elements:
            if element in file.name: exotic_element_crystals.append(file)
            else: None

    for file in files:
        for sym in exotic_sym:
            if sym in file.name: exotic_sym_crystals.append(file)
            else: None

    for file in files: 
        for typ in exotic_types:
            if typ in file.name: exotic_type_crystals.append(file)
            else: None

    #take the interesections of sets to find the unique  
    orig = set(classic_element_crystals).intersection(set(classic_sym_crystals).intersection(classic_type_crystals))
    orig_diff_sym = set(classic_element_crystals).intersection(set(exotic_sym_crystals).intersection(classic_type_crystals))
    orig_diff_ele = set(exotic_element_crystals).intersection(set(classic_sym_crystals).intersection(classic_type_crystals))

    #this is an unnescary 
    exotic = set(exotic_type_crystals)

    #quick sanity check
    if len(orig)+ len(orig_diff_sym) +len(orig_diff_ele) + len(exotic) != len(files):
        print('lengths not equal')
        print(f'files: {len(files)}')
        print(f'orig: {len(orig)} diff_sym: {len(orig_diff_sym)} diff_ele {len(orig_diff_ele)} exotic: {len(exotic)}')
    else: print('All lengths equal')


    #convert to lists
    orig = list(orig)
    orig_diff_sym = list(orig_diff_sym)
    orig_diff_ele = list(orig_diff_ele)
    exotic = list(exotic)
    
    
    return orig, orig_diff_ele, orig_diff_sym, exotic


def create_dataframe_dic(files, step_folders_lst):
    """
    
    
    """
    
    #create my dictionary 
    df_dic = {}
    
    #create the lists in the dictionary and the keys
    for step in step_folders_lst:
        df_dic[f'step_{step}'] = []
        
    #search over all pkls
    for file in files:
        df = pd.read_pickle(file)
        #go through all the steps and append them to corresponding
        #list entry
        for step in step_folders_lst:
            df_dic[f'step_{step}'].append(df.loc[df['step_size'] == str(step)])
            #OLD CODE 
            #lst = df_dic[f'step_{step}']
            #lst.append(df.loc[df['step_size'] == step])
            #df_dic[f'step_{step}'] = lst
    return df_dic
        

def create_step_df_dic(df_dic):
    """
    
    
    """
    
    step_df_dic = {}
    
    for key in df_dic:
        step_df_dic[f'{key}'] = pd.concat(df_dic[key])
        
    return step_df_dic


def create_avg_dic(step_df_dic):
    """
    
    """
    
    #create empty dic
    avg_dic = {}
    
    #stats of interest
    stats = ['key', 'mean', 'std', 'step', '25', '50', '75', 'min', 'max']
   
    #create empty lists to append to
    for stat in stats:
        avg_dic[stat] = []
    
    
    
    for key in step_df_dic:
        
        
        avg_dic['key'].append(key)
        avg_dic['mean'].append(step_df_dic[key].mean())
        avg_dic['std'].append(step_df_dic[key].std())
        avg_dic['step'].append(key)
        avg_dic['25'].append(step_df_dic[key].quantile(q=.25))
        avg_dic['50'].append(step_df_dic[key].quantile(q=.5))
        avg_dic['75'].append(step_df_dic[key].quantile(q=.75))
        avg_dic['max'].append(step_df_dic[key].max())
        avg_dic['min'].append(step_df_dic[key].min())
        
        
          
    return avg_dic
    

def create_stat_dfs(avg_dic):
    """
    
    """
    mean_df = pd.concat(avg_dic['mean'], axis=1).transpose()
    std_df = pd.concat(avg_dic['std'], axis=1).transpose()
    #step_df = pd.concat(avg_dic['key'], axis=1)
    df_25 = pd.concat(avg_dic['25'], axis=1).transpose()
    df_50 = pd.concat(avg_dic['50'], axis=1).transpose()
    df_75 = pd.concat(avg_dic['75'], axis=1).transpose()
    df_min = pd.concat(avg_dic['min'], axis=1).transpose()
    df_max = pd.concat(avg_dic['max'], axis=1).transpose()
    
    return mean_df, std_df, df_25, df_50, df_75, df_min, df_max


def save_dataframes(dfs_lst, dfs_lst_str, crystal_type,
                    emds_folder_name, pandas_path):
    """
    
    
    """
        
    for index, df in enumerate(dfs_lst):
        print(emds_folder_name.name)
        out_name = f'{emds_folder_name.name}_{crystal_type}_{dfs_lst_str[index]}.pkl'
        out_path = pandas_path /out_name
        print(out_path)
        pd.to_pickle(df, out_path)
        
    return  None
     
     
     
     
     
     
     
def sort_noises(files):
    """
    
    """
    no_noise = []
    noise_5 = []
    noise_50 = []
    noise_500 = []
    noise_5000 = []
    
    
    for file in files:
        if "noise_5000" in file.name:
            noise_5000.append(file)
        elif "noise_500" in file.name:
            noise_500.append(file)
        elif "noise_50" in file.name:
            noise_50.append(file)
        elif "noise_5" in file.name:
            noise_5.append(file)
        else:
            no_noise.append(file)
            
            
    return no_noise, noise_5, noise_50, noise_500, noise_5000

def sort_interpolation(files):
    """
    
    """
    ml = []
    lin = []
    

    
    for file in files:
        if "ml" in file.name:
            ml.append(file)
        elif "lin" in file.name:
            lin.append(file)
        else: None
            
            
            
    return ml, lin

def __main__():
    """
    """
    
    ############################################
    ############# Change this stuff ############ 
    ############################################
    #set the emd path
    emd_folder = 'emds_interpolated'
    
    #set what step folders are in there, just the ints
    step_folders = [2, 4, 12, 20]
    ############################################
    
    #set paths 
    base_path = pathlib.Path(sys.argv[1])
    emds_path = base_path / emd_folder
    pandas_path = base_path / 'pandas' / emd_folder
    
    
    #get all pkl files
    files = sorted(emds_path.glob('**/*pkl'))
    
    #sort crystals in to the different types
    orig, orig_diff_ele, orig_diff_sym, exotic = sort_crystals(files)
    
    #list the different crystal types
    crystal_types = [orig, orig_diff_ele, orig_diff_sym, exotic]
    crystal_types_str = ['orig', 'orig_diff_ele', 'orig_diff_sym', 'exotic']
    

       
    #loop over the different crystal types:
    for index, crystal_type in enumerate(crystal_types):

        #get the name of the class
        crystal_type_str = crystal_types_str[index]
        
        print(crystal_type_str)
        print(len(crystal_type))
        
        
        #noises 
        ml, lin = sort_interpolation(crystal_type)
        interpolations = [ml, lin]
        #noises_strings
        interpolations_str = ['ml', 'lin']
        
        for jindex, interpolation in enumerate(interpolations):
            interpolation_str = interpolations_str[jindex]
            
            interpolation_crystal_type_str = crystal_type_str + '_' + interpolation_str
            
            print(interpolation_str)
            print(len(interpolation))
            
            #put all steps into a dictionary with key: step_1 item: [1 row df]... step_n
            df_dic = create_dataframe_dic(interpolation, step_folders)

            #convert into a single dataframe per step
            step_df_dic = create_step_df_dic(df_dic)

            #create a dictionary with list of different averages
            avg_dic = create_avg_dic(step_df_dic)
            
            #create the stat datafames
            mean_df, std_df, df_25, df_50, df_75, df_min, df_max = create_stat_dfs(avg_dic)
            
            #create list of stat dataframes to save
            dfs = [mean_df, std_df] #, df_25, df_50, df_75, df_min, df_max]
            dfs_str = ['mean_df', 'std_df']#, 'df_25', 'df_50', 'df_75','df_min', 'df_max']
            

            save_dataframes(dfs, dfs_str, interpolation_crystal_type_str, emds_path, pandas_path)




if __name__ == "__main__":
    
    __main__()




