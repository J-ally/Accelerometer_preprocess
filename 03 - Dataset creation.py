#!/usr/bin/env python
# coding: utf-8

# # Creation of the dataset

# Processing of all the adjusted annotation files to associate them with their corresponding AX3 files and create a whole dataset.
# 
# Dataset is stored in `/home/Calf Behaviour Classification/PROD/Files/adj_AX3_and_labels_after_synchro`

# # Imports

# In[2]:


import os
import sys  
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date


# ## Path management

# In[1]:


import sys
sys.path.insert(0, '/home/Calf Behaviour Classification/PROD/')
 
# importing the paths management file
from PATHS_management import test_import
 
# Testing the import
if test_import() :
    from PATHS_management import path_adjusted_ax3_folder, path_raw_data_ax3_ref_file, path_adjusted_ax3_with_labels_folder, \
        path_processed_labels_folder,path_boris_to_not_process, \
        path_adjusted_files_NOT_indexed_dataset_file, path_adjusted_files_indexed_dataset_file, \
            path_manual_inspection_output_file, path_dataset_final_folder
            
    print("Import successful")
else :
    print("Import failed")


# ## Functions

# In[3]:


import sys
sys.path.insert(0, '/home/Calf Behaviour Classification/PROD/')
 
# importing the paths management file
from UTILS_data import test_import
 
# Testing the import
if test_import() :
    from UTILS_data import get_closest_ax3_time
    print("Import successful")
else :
    print("Import failed")


# # Creation of the dataset

# ## Imports

# In[4]:


# importing the adjusted boris files ref table
df_raw_data_ref = pd.read_csv(path_raw_data_ax3_ref_file)
df_raw_data_ref.sort_values(by=['calf_id', 'start_date'], inplace=True, ignore_index=True)
display(df_raw_data_ref.head())


# ## Geting all the files for the dataset

# In[5]:


# all the adjusted boris files
all_boris_adjusted_paths = []
for root, dirs, files in os.walk(path_processed_labels_folder):
    for file in files:
        if file.endswith(".csv"):
            all_boris_adjusted_paths.append(os.path.join(root, file))

print(f"all boris adjusted file paths : {all_boris_adjusted_paths}")
print(len(all_boris_adjusted_paths))

# the files to not process PATHS
list_not_to_process_files = [os.path.join(path_boris_to_not_process,f) for f in os.listdir(path_boris_to_not_process) if os.path.isfile(os.path.join(path_boris_to_not_process,f))]

# the files to process NAMES
already_added_to_dataset = []
list_already_processed_files = [os.path.join(path_adjusted_ax3_with_labels_folder,f) for f in os.listdir(path_adjusted_ax3_with_labels_folder) if os.path.isfile(os.path.join(path_adjusted_ax3_with_labels_folder,f))]

for paths in list_already_processed_files :
    name = paths.split('/')[-1]
    name_original = name.split("_")[0] + "_" + name.split("_")[1] + "_" + name.split("_")[2] + ".csv"
    already_added_to_dataset.append(name_original)

already_added_to_dataset.sort()
print(f"file already processed names : {already_added_to_dataset}")
print(len(already_added_to_dataset))

# the files to process PATHS
files_to_process_paths = []
list_boris_processed = [os.path.join(path_processed_labels_folder,f) for f in os.listdir(path_processed_labels_folder) if os.path.isfile(os.path.join(path_processed_labels_folder,f))]

for paths in list_boris_processed :
    name = paths.split('/')[-1]
    name_original = name.split("_")[0] + "_" + name.split("_")[1] + "_" + name.split("_")[2] + ".csv"
    if name_original not in already_added_to_dataset :
        files_to_process_paths.append(f"{path_processed_labels_folder}{name_original.split('.')[0]}_adjusted_short.csv")
        
files_to_process_paths.sort()
print(f"file to process paths : {files_to_process_paths}")
print(len(files_to_process_paths))

# print(len(list_already_processed_files), len(list_not_to_process_files))


# # Synchronization manual inspection correction

# In[6]:


df_manual_inspection = pd.read_csv(path_manual_inspection_output_file)
df_manual_inspection.replace(np.nan, '', inplace=True)
display(df_manual_inspection)


# ## Adding the AX3 data to the boris labels

# In[7]:


# We only add to the dataset the files that have been checked and time adjusted manually ...
nb_files_not_processed = 0

for row in df_manual_inspection.itertuples() :
    
    ##### problem of Nan for the time to add 
    # and even if the time is not added, still need to be part of the dataset, do not put continue there
    if row.nb_seconds_to_add == '' :
        nb_files_not_processed += 1
        continue
        
    file_name = row.file_name
    file_path = f"{path_processed_labels_folder}{file_name}"
    calf_id = int(file_name.split("_")[0])
    camera_start = datetime.strptime(file_name.split("-")[4], '%Y%m%d%H%M%S')
    
    print(f"\n---- {file_name}")
    
    df_boris_adj = pd.read_csv(file_path)
    df_boris_adj['start_adj_datetime'] = pd.to_datetime(df_boris_adj['start_adj_datetime'])
    df_boris_adj['stop_adj_datetime'] = pd.to_datetime(df_boris_adj['stop_adj_datetime'])
    
    # getting the labels and the modifiers together in the same column
    df_boris_adj.replace(to_replace = np.nan, value = '', inplace = True)
    df_boris_adj['label'] = df_boris_adj['label'] + "|" + df_boris_adj['Modifiers']
    # display(df_boris_adj.sort_values("start_adj_datetime").head())
    print(f"Labels and modifiers merged")
    
    if row.nb_seconds_to_add == 0 : # no time to add
        print(f"Zero time added to the boris annotation")
    
    else :
        # adding the time needed to be added to the boris annotation (depending on the manual inspection)
        df_boris_adj["start_adj_datetime"] =df_boris_adj["start_adj_datetime"] + timedelta(seconds=int(row.nb_seconds_to_add))
        df_boris_adj["stop_adj_datetime"] =df_boris_adj["stop_adj_datetime"] + timedelta(seconds=int(row.nb_seconds_to_add))
        # display(df_boris_adj.sort_values("start_adj_datetime").head())
        print(f"Time added to the boris annotation")
            
    # getting the file name and file path of the adjusted AX3 data
    try :
        ax3_file_name , ax3_file_path = get_closest_ax3_time(df_raw_data_ref, calf_id, camera_start)
    except ValueError : # NO adjusted AX3 data found ... should'nt happen at this point, cause checked before
        print(f"\n    ---- {file_name} \nAX3 file not found not existant \n")
        # tracking_file_not_found =  open(path_adjusted_files_NOT_indexed_dataset_file, 'a+')
        # tracking_file_not_found.write(f"\n{file_name}")
        # tracking_file_not_found.close()
        continue
    
    df_adj_acc = pd.read_csv(ax3_file_path, engine='pyarrow')
    df_adj_acc['DateTime'] = pd.to_datetime(df_adj_acc['DateTime'])
    df_adj_acc['adjusted_DateTime'] = pd.to_datetime(df_adj_acc['adjusted_DateTime'])
    # display(df_adj_acc.head())
    print(f"AX3 data loaded")
    
    # Creation of the new dataframe with the adjusted AX3 data and the boris annotation labels
    df_adj_acc_labeled = pd.DataFrame(columns= list(df_adj_acc.columns))
    
    for row in df_boris_adj.itertuples() :
        # if we wanted to cut/add a second or more, uncomment the comments in the following lines
        start = row.start_adj_datetime #+ timedelta(seconds=-1)
        stop = row.stop_adj_datetime #+ timedelta(seconds=-1)
        label = row.label
        
        df_adj_acc_labeled = pd.concat([df_adj_acc_labeled, df_adj_acc.loc[(df_adj_acc['adjusted_DateTime'] >= start) & (df_adj_acc['adjusted_DateTime'] <= stop)].assign(label=label, calf_id=str(calf_id))])

    # saving everything in a csv file
    df_adj_acc_labeled.reset_index(drop=True, inplace=True) # getting rid of the indexes of the AX3 data
    df_adj_acc_labeled.to_csv(f"{path_adjusted_ax3_with_labels_folder}{file_name.split('.')[0]}_labeled.csv", index=False)
    print(f"AX3 labelled in the dataframe : {file_name.split('.')[0]}_labeled.csv")    
    
    # adding the file to the list of processed files
    # tracking_file = open(path_adjusted_files_indexed_dataset_file, 'a+')
    # tracking_file.write(f"\n{file_name.split('/')[-1]}")
    # tracking_file.close()

print(f"--------------------------------------\n LONG individual datasets exported\n--------------------------------------\n")


# # Checking and creating the final dataset

# In[8]:


# all the labelled AX3 files
all_ax3_labelled_paths = []
for root, dirs, files in os.walk(path_adjusted_ax3_with_labels_folder):
    for file in files:
        if file.endswith(".csv"):
            all_ax3_labelled_paths.append(os.path.join(root, file))


# In[17]:


# Simple check
# print(len(all_ax3_labelled_paths), len(df_manual_inspection))

if len (all_ax3_labelled_paths) + nb_files_not_processed != len(df_manual_inspection) :
    raise ValueError("The number of files to process is not correct")
else :
    print("All the files have been correctly processed")


# In[16]:


df_columns = pd.read_csv(all_ax3_labelled_paths[0])
final_dataset = pd.DataFrame(columns= list(df_columns.columns))

for file_path in all_ax3_labelled_paths :
    df = pd.read_csv(file_path, engine='pyarrow')
    final_dataset = pd.concat([final_dataset, df])
    print(f"{file_path.split('/')[-1]} added to the final dataset")

final_dataset.reset_index(drop=True, inplace=True)
final_dataset.to_csv(f"{path_dataset_final_folder}{date.today()}_LONG_dataset.csv", index=False)

print(f"--------------------------------------\n LONG dataset exported\n--------------------------------------\n")

