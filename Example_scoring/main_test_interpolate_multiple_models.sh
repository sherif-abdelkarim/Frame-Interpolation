#!/bin/bash

#set paths hardcoded and relative paths

#HARDCODED
base_path=/home/pattersonlab/Documents/Alex/Tomography_ml/Training_own_models_ab/full/
numpy_path=/home/pattersonlab/Documents/Alex/Tomography_ml/Training_own_models_ab/test_data/npys/
tomviz_pipeline_path=/home/pattersonlab/Documents/Alex/tomviz/tomviz/python/ #path to where piepline path has to run 


echo $base_path
echo $numpy_path
echo $tomviz_pipeline_path

echo 'Setting Relative Paths'
#RELATIVE paths
scripts_path=${base_path}'scripts/' #currently redundant but maybe useful later
emd_path=${base_path}'emds_interpolated/' #path to where all the EMDs will be created 
recon_state_path=${base_path}'Tomviz_files/' #where the tvsm state files are stored 
reconstruction_path=${base_path}'reconstruction/' #where all the reconstructions will go
pandas_path=${base_path}'pandas/' #where the pandas dataframes will go

echo 'ativate conda tf-gpu'
# actiavate conda tf-gpu environment
conda activate tf-gpu


echo 'Moving to Base'
#move to base path
cd $base_path

echo 'Making emd folder'
#make emd folder
mkdir $emd_path

echo 'Normalising numpy stacks'
#normalise the numpy stacks between [0,1]
python3 ${scripts_path}Normalise_numpys.py ${numpy_path}

echo 'Converting numpys to emd and saving them in emd folder'
#convert numpys to emd and save them in the emd folder 
python3 ${scripts_path}npy_to_emd_v8.py $numpy_path $emd_path 

echo 'Creating a parent folder each crystal'
#create a parent folder for each crystal
python3 ${scripts_path}making_parent_crystals_folders.py ${emd_path}

echo 'Making step folders [2..20]'
#make step folders for each crystal
python3 ${scripts_path}make_step_folders_multiple_models.py ${emd_path}

echo 'Moving emds to corresponding folders'
#move the emds in to the parent folders
python3 ${scripts_path}move_emds.py ${emd_path}

echo 'Creating all the differnt step size stacks'
#make all the different step cyrstals [2..20]
python3 ${scripts_path}make_all_step_stack_multiple_models.py ${emd_path}

echo 'Making interpolated stacks'
# Make the interpolated stacks 
python3 ${scripts_path}make_interpolated_stacks_multiple_models.py ${emd_path} ${base_path}

echo 'Converting all the different step size numpys as emds' 
#I have to clean up and run all npy to emd script as I made a mistake in the order i ran stuff
python3 ${scripts_path}npy_to_emd_v8_2.py ${emd_path}

echo 'Moving to tomviz-pipeline path'
#Change to the tomviz-pipeline pathway
cd ${tomviz_pipeline_path}

echo 'stop using tf-gpu environment'

conda deactivate


echo 'Running the reconstructions'
#reconstruction 
python3 ${scripts_path}recon_pipeline_inter_multiple_models.py ${base_path}

echo 'Moving to base path'
#move to base path
cd ${base_path}

echo 'Creating a dataframe for each crystal'
python3 ${scripts_path}consolidate_crytals_emds_interpolated_multiple_models.py ${base_path}

pthon3 ${scripts_path}create_average_dataframes_emds_interpolation_multiple_models.py ${base_path}



echo 'Done'
