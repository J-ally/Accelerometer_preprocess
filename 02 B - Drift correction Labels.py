#!/usr/bin/env python
# coding: utf-8

# # Documentation
# 
# The whole process relies on two different files `df_adjusting_data_tracking.csv` as well as `df_adjusting_plot_tracking.csv`. Both of these files are the references which stores :
# - which files has already been processed from the `Labelling_Files` folder.
# - where the files that has been processed have been exported to.
# 
# The key identifier of each file is the `file_name` column. This column is the name of the file with the extension : it refers to the original file name that is first uploaded in the `Labelling_Files` folder. This path is stored as well for convenience as the `file_path`.
# 
#     For example, if the file is `Calf_1_2019-11-01_2019-11-30.csv`
#     `file_name` will be : `Calf_1_2019-11-01_2019-11-30.csv`
#     `file_path` would be : /home/admin/Calf_Video_labelling/Labelling_Files/Oshana/Calf_1_2019-11-01_2019-11-30.csv`.
# 
# The `df_adjusting_data_tracking.csv` is structured this way :
# 
# | file_path    	| file_name 	| adjusted 	| date_time_ref_start 	| date_time_ref_stop  	| date_time_camera_start 	| 
# |-------------------------------------------------------------------------------------------------------------------------	|-----------------------------------------------------------------	|----------	|---------------------	|---------------------	|------------------------	|
# | ..../1308_record-0000-0001-CH02-20220221160000-20220221170000_LR.csv 	| 1308_record-0000-0001-CH02-20220221160000-20220221170000_LR.csv 	| True     	| 2022-02-18 16:03:16 	| 2022-02-25 15:28:02 	| 2022-02-18 16:03:23    	| 
# 
# 
# 
# | date_time_camera_stop 	| camera_start_time_adjusted 	| calf_id 	| output_path_lab_full   	| output_path_lab_short      	|
# |----------------------|----------------------------|--------------|--------|---------------	|
# | 2022-02-25 15:28:22   	|2022-02-21 15:59:47.500000 	| 1308    	| ..../Raw Boris Adjusted Times/1308_record-0000-0001-CH02-20220221160000-20220221170000_LR_adjusted.csv 	| ..../Raw Boris Adjusted Times/1308_record-0000-0001-CH02-20220221160000-20220221170000_LR_adjusted_short.csv 	|
# 
# - adjusted : True if the file has been processed, False if not.
# - date_time_ref_start : the closest reference time (AX3 data) before the date of the start of the video
# - date_time_ref_stop : the closest reference time (AX3 data) after the date of the start of the video
# - date_time_camera_start : the closest camera time before the start of the video
# - date_time_camera_stop : the closest camera time after the start of the video
# 
# ```note
# 
#      AX3 start (date_time_ref_start)                                    AX3 stop (date_time_ref_stop)
#          |------------------------------------------------------------------|
# Video start (date_time_camera_start)                                          Video stop (date_time_camera_stop)
#     |----------------------------------------------------------------------------|
# 
#                      |----| Video we are interested in
# ```
# 
# - camera_start_time_adjusted : the start camera time adjusted (frame wise)
# - calf_id : the calf id
# - output_path_lab_full : the path where the file has been exported to (full file)
# - output_path_lab_short : the path where the file has been exported to (short file), can be directly used for the ML process.
# 
# 
# The `df_adjusting_plot_tracking.csv` is structured this way :
# 
# | file_path    	| file_name 	| calf_id 	| adjusted	| ploted  	| output_path_plot 	| output_path_pkl |
# |---------------------------	|-------------------------------------------------	|----------	|---------------------	|---------------------	|------------------------	|-------|
# | ..../1308_record-0000-0001-CH02-20220221160000-20220221170000_LR.csv 	| 1308_record-0000-0001-CH02-20220221160000-20220221170000_LR.csv | 1308	| True     	| True 	| None	| None    	| 
# 
# 
# - calf_id : the calf id
# - adjusted : True if the file has been processed, False if not.
# - ploted : True if the file has been ploted, False if not.
# - output_path_plot : the path where the plot has been exported to. If the cannot be plotted because there isn't any AX3 data associated with the video, the value is None.
# - output_path_pkl : the path where the pkl file has been exported to. If the cannot be plotted because there isn't any AX3 data associated with the video, the value is None.
# 
# 
# # Outputs
# 
# This code generate 4 diffferent outputs for each new file processed :
# 
# - a full annotation file with the adjusted times (full file) - saved at `output_path_lab_full`
# - a short annotation file with the adjusted times (short file) ready to use for any ML applications - saved at `output_path_lab_short`
# - a plot of the original annotation file with the adjusted times (full file) - saved at `output_path_plot`
# - a pickle file of the plot associated with the file for future use if needed - saved at `output_path_pkl`
# 

# # Path Management and imports

# In[1]:


import pandas as pd
import numpy as np
from datetime import timedelta
import pickle as pkl

import time
import glob, os
import matplotlib.pyplot as plt


# In[2]:


import sys
sys.path.insert(0, '/home/Calf Behaviour Classification/PROD/')
 
# importing the paths management file
from PATHS_management import test_import
 
# Testing the import
if test_import() :
    from PATHS_management import path_output_adj_annotations_folder, path_output_boris_adjusted_folder,path_processed_labels_folder, \
        path_boris_to_not_process, \
        path_output_figs_ax3_folder, path_output_pkl_ax3_folder, path_available_data_file,path_parent_all_labelling_folder, \
        path_already_adjusted_plots_file, path_raw_data_ax3_ref_file
        #path_already_adjusted_files_file, 
        
else :
    print("Import failed")


# In[3]:


cam_ref_times = pd.read_csv(path_available_data_file, usecols=['pen','camera_datetime','reference_datetime', "gap"])
cam_ref_times['camera_datetime'] = pd.to_datetime(cam_ref_times['camera_datetime'], format='%Y-%m-%d %H:%M:%S')
cam_ref_times['reference_datetime'] = pd.to_datetime(cam_ref_times['reference_datetime'], format='%Y-%m-%d %H:%M:%S')
cam_ref_times.sort_values(by=['pen','camera_datetime'], inplace=True)
display(cam_ref_times.head())


# In[4]:


# importing the adjusted boris files ref table
df_raw_data_ref = pd.read_csv(path_raw_data_ax3_ref_file)
df_raw_data_ref.sort_values(by=['calf_id', 'start_date'], inplace=True, ignore_index=True)
display(df_raw_data_ref.head())


# In[5]:


# The color for the behaviours

# other colors = [ 'tomato', 'chocolate', 'chartreuse', 'deepskyblue','cyan', 'mediumorchid']
behavior_groups = {'lying' : {"group":[ 'lying'], "color" : "purple"},
                 'transition' : {"group" : ['lying-down',"rising"], "color" : "pink"},
                 'standing': {"group" : ["standing", "SRS"], "color" : "yellow"}, 
                 'walking': {"group" : ["walking", "sniff"], "color" : "blue"},
                 'eating': {"group" : ["drinking", "eating", "rumination", "defecation", "urination"], "color" : "green"}, 
                 'grooming': {"group" : ["grooming"], "color" : "orange"}, 
                 'play': {"group" : ["run", "play", "social interaction"], "color" : "red"}, 
                 'abnormal': {"group" : ["abnormal", "oral manipulation of the pen", "oral manipulation of pen", "cough"], "color" : "black"}
                }


# # Managing the boris annotation files

# ## Functions for the boris annotation files

# In[6]:


def get_data_from_raw_data_adjusted_path (file_path : str) -> pd.Timestamp :
    """
    returns the calf id, start and end date of the raw data file

    Args:
        file_path (str): the name/path of the file

    Returns:
        start_ax3_adjusted (pd.Timestamp): the start date of the raw data file
        end_ax3_adjusted (pd.Timestamp): the end date of the raw data file 
    """
    #/home/Calf Behaviour Classification/PROD/Files/raw_data_per_shake_rounds/1302_2022-01-21T11:55:05_2022-02-04T09:34:00.csv
    file_name = file_path.split("/")[-1]
    calf_id = file_name.split("_")[0]
    start_date = file_name.split("_")[1]
    end_date = file_name.split("_")[2].split(".")[0]
    start_ax3_adjusted = pd.to_datetime(start_date, format="%Y-%m-%dT%H:%M:%S")
    end_ax3_adjusted = pd.to_datetime(end_date, format="%Y-%m-%dT%H:%M:%S")
    
    return calf_id, start_ax3_adjusted, end_ax3_adjusted
        
        
def get_start_date_from_boris_filename (file_path : str) -> pd.Timestamp :
    """
    returns the start date of the video from the filename

    Args:
        file_path (str):  the name of the file
    Returns:
        start_date (pd.Timestamp): the date of the start of the video
    """
    #/home/admin/Calf_Video_labelling/Labelling_Files/Lucile/1308_record-0000-0001-CH02-20220221160000-20220221170000_LR.csv
    name_file = file_path.split("/")[-1]
    name_file_start_date = name_file.split("-")[-2]
    start_date = pd.to_datetime(name_file_start_date, format="%Y%m%d%H%M%S") 
    return start_date


def get_pen_number_from_boris_filename (file_path : str) -> int :
    """
    returns the pen number from the filename

    Args:
        file_path (str): the name/path of the file
    Returns:
        pen_number (int): the pen number of the file
    """
    #/home/admin/Calf_Video_labelling/Labelling_Files/Lucile/1308_record-0000-0001-CH02-20220221160000-20220221170000_LR.csv
        
    camera = file_path.split("-")[-3]
    start_date = get_start_date_from_boris_filename(file_path)
    if start_date < pd.Timestamp('2022-02-18 00:00:00') and camera == "CH01" :
        pen_number = 2
    elif start_date < pd.Timestamp('2022-02-18 00:00:00') and camera == "CH02" :
        pen_number = 3
    else :
        pen_number = int(camera[-1])
    return pen_number


def get_closest_ref_time (ref_table : pd.DataFrame, pen_number : int, ref_date : pd.Timestamp) :
    """
    returns the ref times that frames the given the pen number and the date

    Args:
        ref_table (pd.DataFrame): table with all the camera and ref times
            Table needs to have the following columns: pen, camera_datetime, reference_datetime
        pen_number (int): the pen number of the calf (1,2,3,4)
        ref_date (pd.Timestamp) : time in the format %Y-%m-%d %H:%M:%S
        
    Returns:
        reference_datetime_1 (pd.Timestamp) : the closest reference time before the given date
        reference_datetime_2 (pd.Timestamp) : the closest reference time after the given date
        camera_datetime_1 (pd.Timestamp) : the closest camera time before the given date
        camera_datetime_2 (pd.Timestamp) : the closest camera time after the given date
    """
    # print(ref_date)
    # display(ref_table)
    time_distances = []
    searched_table = ref_table[ref_table['pen']==pen_number]
    searched_table_ordered = searched_table.sort_values(by=['reference_datetime'])
    for row in searched_table_ordered.itertuples() :
        time_distances.append( ( (row.reference_datetime - ref_date).total_seconds(), row.Index))
    
    time_distances.sort(key=lambda x: x[0])
    min_time_dist, min_time_index = min(time_distances, key= lambda x: abs(x[0]))
    # print(min_time_dist)
    
    if min_time_dist < 0 :
        time_distances_pos = [x for x in time_distances if x[0] > 0]
        min_time_dist_2, min_time_index_2 = time_distances_pos[0]
        # print(time_distances)
        return searched_table_ordered.loc[min_time_index, 'reference_datetime'], searched_table_ordered.loc[min_time_index_2, 'reference_datetime'], \
        searched_table_ordered.loc[min_time_index, 'camera_datetime'], searched_table_ordered.loc[min_time_index_2, 'camera_datetime']
        
    else :
        time_distances_neg = [x for x in time_distances if x[0] < 0]
        min_time_dist_2, min_time_index_2 = time_distances_neg[0]
        return searched_table_ordered.loc[min_time_index_2, 'reference_datetime'], searched_table_ordered.loc[min_time_index, 'reference_datetime'], \
        searched_table_ordered.loc[min_time_index_2, 'camera_datetime'], searched_table_ordered.loc[min_time_index, 'camera_datetime']

def generating_df_frames (pen_number : int, date_time_camera_start : pd.Timestamp, date_time_camera_stop : pd.Timestamp, date_time_ref_start : pd.Timestamp, date_time_ref_stop : pd.Timestamp) : 
    """
    First check if the frames id are already generated, if not,
    generates a dataframe with the frames id and the corresponding times and saves it 
    in a CSV file at : /home/admin/Calf_Video_labelling/Labelling Files gestion/Frame Index Files

    Args:
        pen_number (int): the pen number of where the calf is (1,2,3,4)
        date_time_camera_start (pd.Timestamp): startin time of the camera
        date_time_camera_stop (pd.Timestamp): stopping time of the camera
        date_time_ref_start (pd.Timestamp): starting time of the reference
        date_time_ref_stop (pd.Timestamp): stopping time of the reference

    Returns:
        df_time_correspondance (df): the dataframe with the frames id and the corresponding times
    """
    saving_path = "/home/admin/Calf_Video_labelling/Labelling Files gestion/Frame Index Files"
    # checking if the file already exists
    for file in os.listdir(saving_path) :
        if file == f"pen_{pen_number}_start_{date_time_ref_start}_stop_{date_time_ref_stop}_frames_index.csv" :
            df_time_correspondence = pd.read_csv(f"{saving_path}/{file}") 
            df_time_correspondence['date_time_camera_theorical'] = pd.to_datetime(df_time_correspondence['date_time_camera_theorical'], format='%Y-%m-%d %H:%M:%S')
            df_time_correspondence['date_time_camera_observed'] = pd.to_datetime(df_time_correspondence['date_time_camera_observed'], format='%Y-%m-%d %H:%M:%S')
            df_time_correspondence['date_time_ref'] = pd.to_datetime(df_time_correspondence['date_time_ref'], format='%Y-%m-%d %H:%M:%S')
            
            return df_time_correspondence
        
    # Calculating the number of frames
    tot_sec = (date_time_ref_stop - date_time_ref_start).total_seconds()
    fs = 8
    N = int(tot_sec * fs)
    # print("Number of frames: ", N)

    # aligning the times
    diff_ref_camera_start = pd.to_timedelta((date_time_ref_start  - date_time_camera_start), unit="s")

    date_time_camera_start_aligned = date_time_camera_start + diff_ref_camera_start
    date_time_camera_stop_aligned = date_time_camera_stop  + diff_ref_camera_start

    # Calculating the drift
    drift_tot =  date_time_ref_stop - date_time_camera_stop_aligned # drift always calculated in comparison to the ref time (so ref time - camera time)
    drift_frame = drift_tot/tot_sec/fs # if drift < 0 --> camera time is in advance compared to ref time (ex. camera time = 2022-02-25 15:28:34 and ref time = 2022-02-25 15:28:21)

    # Constructing the df
    frame_id = list(np.arange(0,N+1))
    date_time_camera_theorical = [date_time_camera_start + pd.to_timedelta(frame_id[x]*1/fs,unit= "s") for x in frame_id]

    df_time_correspondence = {"id_frame" : frame_id, "date_time_camera_theorical" : date_time_camera_theorical}
    df_time_correspondence = pd.DataFrame(df_time_correspondence)

    df_time_correspondence["date_time_camera_observed"] = df_time_correspondence.apply(lambda x: x["date_time_camera_theorical"] - drift_frame*x.id_frame, axis=1) # in this case we want to add the drift to the theorical camera time --> substract the drift
    df_time_correspondence["date_time_ref"] = [date_time_ref_start + pd.to_timedelta(frame_id[x]*1/fs,unit= "s") for x in frame_id]
    
    # saving the new correspondence file
    df_time_correspondence.to_csv(f"{saving_path}/pen_{pen_number}_start_{date_time_ref_start}_stop_{date_time_ref_stop}_frames_index.csv", index=False)
    
    return (df_time_correspondence)


def load_raw_boris_file (path_boris : str, display_df : bool = True ) -> pd.DataFrame:
    """
    load a boris file and returns the dataframe

    Args:
        path_boris (str): the path to the boris file
        display (bool): if True, displays the first 5 rows of the dataframe
    Returns:
        labels (pd.DataFrame): the dataframe coming with the boris file
    """
    
    labels = pd.read_csv(path_boris)
    # display(labels.tail)
    
    start_date = get_start_date_from_boris_filename(path_boris)
    
    labels["Observation date"] = start_date    
        
    labels['Start (s)'] = pd.to_timedelta(labels['Start (s)'], unit='s')
    labels['Stop (s)'] = pd.to_timedelta(labels['Stop (s)'], unit='s')
    
    if display_df :
        display(labels.head())
        
    return labels


def adjust_boris_labelling_time (df_raw_boris : pd.DataFrame, new_video_start_time : pd.Timestamp, display_df : bool = True) -> pd.DataFrame:
    """
    change the time of the labelling to match the adjusted time of the video

    Args:
        df_raw_boris (pd.DataFrame): the dataframe coming with imported the boris file
        new_video_start_time (pd.Timestamp): the new starting time of the video obtained from the frame index file
        display_df (bool): if True, displays the first 5 rows of the dataframe
    Returns:
        pd.DataFrame : the new boris dataframe with the adjusted times
    """
    labels = df_raw_boris.copy()
    
    labels['Observation date'] = new_video_start_time
    
    labels['start_adj_datetime'] = labels['Start (s)'] + labels['Observation date']
    labels['stop_adj_datetime'] = labels['Stop (s)'] + labels['Observation date']
    
    if display_df :
        display(labels.head())
        
    return (labels)


# In[7]:


# testing some of the functions
print(get_data_from_raw_data_adjusted_path("/home/Calf Behaviour Classification/PROD/Files/raw_data_per_shake_rounds/1302_2022-01-21T11:55:05_2022-02-04T09:34:00.csv"))
# print(get_closest_ref_time(cam_ref_times, 1, pd.Timestamp('2022-03-06 16:26:40')))
test_boris= load_raw_boris_file("/home/admin/Calf_Video_labelling/Labelling_Files/Lucile/1308_record-0000-0001-CH02-20220221160000-20220221170000_LR.csv")
print(get_pen_number_from_boris_filename("/home/admin/Calf_Video_labelling/Labelling_Files/Lucile/1308_record-0000-0001-CH02-20220121160000-20220221170000_LR.csv"))


# ## Getting the already existing files

# In[8]:


# getting in the files in the 'processed_labels_files' folder only
list_processed_files = [os.path.join(path_processed_labels_folder,f) for f in os.listdir(path_processed_labels_folder) if os.path.isfile(os.path.join(path_processed_labels_folder,f))]
print(f"file processed paths : {list_processed_files}")
print(len(list_processed_files))

# getting the original name of the files
list_processed_files_name = []
for paths in list_processed_files :
    name = paths.split("/")[-1]
    original_name = name.split("_")[0] + "_" + name.split("_")[1] + "_" + name.split("_")[2] + ".csv"
    list_processed_files_name.append(original_name)
    
print(list_processed_files_name)

# getting in the files in the 'processed_labels_files/To check' folder only
files_to_not_process = [os.path.join(path_boris_to_not_process,f) for f in os.listdir(path_boris_to_not_process) if os.path.isfile(os.path.join(path_boris_to_not_process,f))]
                
# getting the original name of the files
list_to_not_process_files_name = []
for paths in files_to_not_process :
    name = paths.split("/")[-1]
    original_name = name.split("_")[0] + "_" + name.split("_")[1] + "_" + name.split("_")[2] + ".csv"
    list_to_not_process_files_name.append(original_name)
    
print(f"file not to process paths : {files_to_not_process}")
print(len(files_to_not_process))
print(list_to_not_process_files_name)


# ## Getting the new files

# In[9]:


all_labelling_paths = []
for root, dirs, files in os.walk(path_parent_all_labelling_folder):
    for file in files:
        if file.endswith(".csv"):
            all_labelling_paths.append(os.path.join(root, file))

print(f"all label file paths : {all_labelling_paths}")

files_to_process_paths = []

for files_paths in all_labelling_paths :
    name = files_paths.split("/")[-1]
    if name not in list_to_not_process_files_name :
        if name not in list_processed_files_name :
            files_to_process_paths.append(files_paths)

print(f"label file paths to process : {files_to_process_paths}")
print(len(files_to_process_paths))


# ## Processing the raw boris files

# In[21]:


for file_path in files_to_process_paths[:] :
    
    file_name = file_path.split("/")[-1]
    
    if file_name in list_processed_files_name :
        print(f"-----  video {file_name} \
            \nfile '{file_path}'end= already processed \
            \nAdjusted boris file_path generated and exported\n")
        continue
    
    # Get the boris file_path 
    raw_boris_file = load_raw_boris_file(file_path, display_df = False)
    
    # Get the infos
    raw_boris_observation_date = raw_boris_file["Observation date"].values[0]
    calf_id = int(raw_boris_file["Subject"].values[0])
    pen_number = get_pen_number_from_boris_filename(file_path)
    camera_start_time = get_start_date_from_boris_filename(file_path)

    # Get closest ref time
    date_time_ref_start, date_time_ref_stop, date_time_camera_start, date_time_camera_stop = get_closest_ref_time(cam_ref_times, pen_number, camera_start_time) 
    
    # Get the correspondence between the frame and the time
    df_time_correspondence = generating_df_frames(pen_number, date_time_camera_start, date_time_camera_stop, date_time_ref_start, date_time_ref_stop)

    date_time_start_video_adjusted = df_time_correspondence.date_time_ref[df_time_correspondence["date_time_camera_observed"]>=camera_start_time].iloc[0]

    # Adjust and export the boris file_path
    df_boris_adjusted = adjust_boris_labelling_time(raw_boris_file, date_time_start_video_adjusted, display_df = False)
    
    export_name = file_name.split(".")[0] #remove the .csv
    df_boris_adjusted.to_csv(f"{path_output_boris_adjusted_folder}{export_name}_adjusted.csv", index = False)

    try : #normal
        df_boris_short = df_boris_adjusted[['Observation id', 'Observation date', 'Subject', 'Behavior','Modifiers','start_adj_datetime','stop_adj_datetime']].copy()
    except KeyError : # Oshana's file_path, no "Modifiers" column
        df_boris_short = df_boris_adjusted[['Observation id', 'Observation date', 'Subject', 'Behavior','start_adj_datetime','stop_adj_datetime']].copy()
        
        try : # If there is a third modifier column
            df_boris_short.insert(loc = 4, column="Modifiers", value = df_boris_adjusted['Modifier #1'] + "|" + df_boris_adjusted['Modifier #2']+"|"+df_boris_adjusted['Modifier #3'])
        except KeyError : # no third modifier column
            df_boris_short.insert(loc = 4, column="Modifiers", value = df_boris_adjusted['Modifier #1'] + "|" + df_boris_adjusted['Modifier #2'])
            
    df_boris_short.rename(columns={'Observation id' : 'observation_id','Observation date':'observation_date','Subject' : 'calf_id', 'Behavior': 'label'}, inplace=True)
    df_boris_short.to_csv(f"{path_processed_labels_folder}{export_name}_adjusted_short.csv", index = False)

    # Updates all the tables
 
    print(f"-----  video {file_name} \
        \noriginal video start time : {camera_start_time} \
        \ndate_time_start_video_adjusted : {date_time_start_video_adjusted} \
        \nAdjusted boris file_path generated and exported \
        \nShortened Boris file_path generated and exported\n")

print(f"--------------------------------------\n Boris file treated and exported\n--------------------------------------\n")


# # Managing the plots for testing

# ## Functions for the ploting part

# In[14]:


def filter_behaviour_and_get_color (behavior_grouping : dict, current_behavior : str) :
    """
    Filter the dataframe to only keep the behaviours we want to analyse and get the color associated 
    to each behaviour for plotting
    
    Args:
        behavior_grouping (dict): dictionary containing the behaviours we want to analyse (and their grouping) 
    and the color associated to each behaviour. Must have a structure like this :
    {'behaviour1' : {'group' : 'group1', 'color' : 'color1'}, 'behaviour2' : {'group' : 'group2', 'color' : 'color2'}, ...}
        current_behaviour (str): current behaviour we are analysing
    
    Returns:
        color (str) : color associated to the current_behaviour 
        label (str) : filtered label associated with the current behaviour
        
    Comments:
        E.g. : color, label = filter_behaviour_and_get_color (behavior_groups, 'walking')
    """

    for key, value in behavior_grouping.items() :
        if current_behavior == key :
            color = value['color']
            label = key
            return color, label
        else :
            if current_behavior in behavior_grouping[key]['group'] :
                color = value['color']
                label = key
                return color, label
                
    return "Behavior not in the dictionary ..."


def get_closest_ax3_time (df_ax3_adj_ref : pd.DataFrame, calf_id : int, ref_date : pd.Timestamp) -> str :
    """
    returns the name of the ax3 times file that frames the ref date given the calf_id
       ===> the file that contains the ax3 values for the given ref date and calf id

    Args:
        df_ax3_adj_ref (pd.DataFrame): table with all the camera and ref times
            Table needs to have the following columns: pen, camera_datetime, reference_datetime
        calf_id (int): the id of the calf
        ref_date (pd.Timestamp) : time in the format %Y-%m-%d %H:%M:%S
        
    Returns:
        file_name (str) : the name of the ax3 time that frames the ref date
        file_path (str) : the path of the ax3 time that frames the ref date
    """
    
    searched_table = df_ax3_adj_ref[df_ax3_adj_ref['calf_id']==calf_id]
    searched_table_ordered = searched_table.sort_values(by=['start_date'], ignore_index=True)
    searched_table_ordered['end_date'] = pd.to_datetime(searched_table_ordered['end_date'])
    searched_table_ordered['start_date'] = pd.to_datetime(searched_table_ordered['start_date'])

    for row in searched_table_ordered.itertuples() :
        if row.start_date < ref_date and row.end_date > ref_date :
            file_name = searched_table_ordered.iloc[row.Index].file_name
            file_path = searched_table_ordered.iloc[row.Index].file_path
            return file_name, file_path
                    
    return "No File Name Found"
    

def plot_and_save_magn (df_labels : pd.DataFrame, df_acc : pd.DataFrame, calf_id : int, behavior_groups : dict,file_name, output_figs : str,output_pkl : str, figsize_ = (14, 10)) :
    
    FONTSIZE = 14
    get_ipython().run_line_magic('matplotlib', 'widget')
    plt.style.use('ggplot')
    fig_handle = plt.figure(figsize = figsize_)
    
    all_behaviours = df_labels.Behavior.unique()
    
    time_pattern_inf = df_labels["Observation date"].values[0]
    time_pattern_sup = time_pattern_inf + pd.Timedelta(minutes = 60)
    
    acc_plot = df_acc.loc[df_acc["adjusted_DateTime"].between(time_pattern_inf, time_pattern_sup)].iloc[np.arange(0,len(df_acc.Amag.loc[df_acc["adjusted_DateTime"].between(time_pattern_inf, time_pattern_sup)]))]
    plt.plot(acc_plot.adjusted_DateTime, acc_plot.Amag, color =  "#0F6C7B", label = "other")

    # getting the Amag corresponding to the labels
    for row in df_labels.itertuples() :
        time_pattern_inf = row[-2]
        time_pattern_sup = row[-1]
        df_acc_plot = df_acc.loc[df_acc["adjusted_DateTime"].between(time_pattern_inf, time_pattern_sup)].iloc[np.arange(0,len(df_acc.Amag.loc[df_acc["adjusted_DateTime"].between(time_pattern_inf, time_pattern_sup)]))]
        adj_datetime_i = df_acc_plot.adjusted_DateTime
        amag_i = df_acc_plot.Amag
        
        try :
            my_color, my_label = filter_behaviour_and_get_color(behavior_grouping = behavior_groups, current_behavior= row.Behavior)
        except ValueError :
            print(f"Behavior '{row.Behavior}' not in the behavior groups dictionary ...")
            raise ValueError
        
        plt.plot(adj_datetime_i, amag_i, color = my_color, label = my_label)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    plt.ylim(0, 2)
    plt.legend(by_label.values(), by_label.keys())
    plt.title(f"Magnitude of the acceleration - calf {int(calf_id)} \n with the following labels : {all_behaviours} \
        \n from the file {file_name}",fontsize=FONTSIZE)
    plt.xlabel('Adjusted DateTime',fontsize=FONTSIZE)
    plt.ylabel('Magnitude',fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=8)
    # plt.show()
    
    plt.savefig(f"{output_figs}{file_name}_plot.png")
    
    file = open(f"{output_pkl}{file_name}_plot.pickle", "wb")
    pkl.dump(fig_handle, file)
    file.close()    
    
    plt.close()
    
    
def loading_plot (path_to_pkl : str) :
    """
    loading a plot from a pickle file
    """
    get_ipython().run_line_magic('matplotlib', 'widget')
    fig_handle = pkl.load(open(path_to_pkl, "rb"))
    fig_handle.show()
    return 


# ## Getting the already existing plots

# In[15]:


# getting in the files in the 'processed_labels_files' folder only
list_processed_plots_paths = [os.path.join(path_output_figs_ax3_folder,f) for f in os.listdir(path_output_figs_ax3_folder) if os.path.isfile(os.path.join(path_output_figs_ax3_folder,f))]
print(f"file processed plots paths : {list_processed_plots_paths}")
print(len(list_processed_plots_paths))

#/home/admin/Calf_Video_labelling/Labelling Files gestion/Labels Figures Testing Output/1302_record-0000-0001-CH02-20220221170000-20220221180000_OI_plot.png

# getting the original name of the files
list_processed_plots_name = []
for paths in list_processed_plots_paths :
    name = paths.split("/")[-1]
    original_name = name.split("_")[0] + "_" + name.split("_")[1] + "_" + name.split("_")[2] + ".csv"
    list_processed_plots_name.append(original_name)

print(f"file processed plots name: {list_processed_plots_name}")
print(len(list_processed_plots_name))


# ## Getting the files to plot

# In[16]:


files_to_plot_paths = []

for files_paths in all_labelling_paths :
    name = files_paths.split("/")[-1]
    # print(name)
    if name not in list_to_not_process_files_name :
        if name not in list_processed_plots_name :
            files_to_plot_paths.append(files_paths)

print(f"plot file paths to process : {files_to_plot_paths}")
print(len(files_to_plot_paths))


# ## Processing the plots from the raw boris files

# In[17]:


for file_path in files_to_plot_paths[:] :

    file_name = file_path.split("/")[-1]
    file_name = file_name.split(".")[0] #remove the .csv
    
    if file_name in list_processed_plots_name :
        print(f"-----  video {file_name}.csv \
            \nfile '{file_path}'end= already processed \
            \nAdjusted boris file_path generated and exported\n")
        continue
    
    original_file_name = file_name.split("_")[0] + "_" + file_name.split("_")[1] + "_" + file_name.split("_")[2]
    
    processed_short_boris_file_name = original_file_name + "_adjusted_short.csv"
    processed_boris_file_name = original_file_name + "_adjusted.csv"
    
    # get the camera adjusted start time
    try :
        processed_boris_file = pd.read_csv(f"{path_processed_labels_folder}{processed_short_boris_file_name}")
        camera_start = processed_boris_file["observation_date"].values[0]
        camera_start = pd.to_datetime(camera_start)
    except ValueError :
        print(f"---- {file_name} \nBoris file not found not existant \n")
        continue

    calf_id = int(file_name.split("_")[0])
    
    try :
        file_name_i , file_path_i = get_closest_ax3_time(df_raw_data_ref, calf_id, camera_start)
        
    except ValueError :
        print(f"---- {original_file_name} \nAX3 file not found not existant \n")
        continue
        
    df_adj_acc = pd.read_csv(file_path_i, engine='pyarrow')
    df_adj_acc['DateTime'] = pd.to_datetime(df_adj_acc['DateTime'])
    df_adj_acc['adjusted_DateTime'] = pd.to_datetime(df_adj_acc['adjusted_DateTime'])
    
    path_boris_adjusted = f"{path_output_boris_adjusted_folder}{processed_boris_file_name}"
    
    df_labels = pd.read_csv(path_boris_adjusted)
    df_labels['Observation date'] = pd.to_datetime(df_labels['Observation date'])
    labels_before_filter = df_labels.Behavior.unique()
    
    plot_and_save_magn(df_labels, df_adj_acc, int(calf_id), behavior_groups, original_file_name, path_output_figs_ax3_folder, path_output_pkl_ax3_folder)
    
    print(f"---- {original_file_name} \
        \ninitial labels : {labels_before_filter} \
        \nplot, pickle saved in {path_output_figs_ax3_folder} \n")

print(f"--------------------------------------\n df_adjusting_plot_tracking.csv exported\n--------------------------------------\n")
    

