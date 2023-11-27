#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# Correcting the time stamps of Raw Accelerometer data based on the identified shake launch and stoptimes by the Shake Pattern Detection Algorithm.
# Time Correction Process Steps :
# 1. Synchronise the launch time (get the gap between the ref_launch_datetime and the algo_launch_datetime and reduce that gap from every timestamp so the ref_launch_datetime = algo_launch_datetime)
# 
# 2. Calculate the drift = (get the difference between the ref_stop_datetime - algo_stop_datetime)
# 
# 3. Calculate the drift per entry = drift / number of entries in the dataframe 
# (the drift per entry is calculated so the drift can be applied linearly on the dataframe. By this way, the final correction would be drift_per_entry x last_index = drift, thus: ref_stop_datetime = algo_stop_datetime)
# 
# 4. Apply the correction to the dataframe
# 
# # Inputs
# 
# The dataframe constructed in the previous step, stored with the name `algo_data_available_for_complete_shake_rounds.csv`*
# It has the following architecture : 
# | calf_id | round_trial | ref_launch_datetime | ref_stop_datetime | algo_launch_datetime  | algo_stop_datetime |
# |---------|-------------|---------------------|-------------------|-----------------------|--------------------|
# |  1      |      1      | 2022-01-21 11:55:05 | 2022-02-04 09:34:00  | 2022-01-21 11:55:06 | 2022-02-04 09:34:01 |
# |  ...      |      ...      | ...| ... | ... | ... |
# 
# /!\ Time corrections can only be applied to complete shake rounds. The algorithm must detect both the launch and end shakes of a shake round to provide the necessary information for calculating drift in the accelerometer data.
# 
# # Outputs
# 
# The dataframes with the corrected timestamps, stored with the name `corrected_algo_data_available_for_complete_shake_rounds.csv`
# 
# Updates the table containing all the reference times for the different shake rounds, stored at `/home/Calf Behaviour Classification/PROD/Files/AX3_adjusted_per_shake_round/Reference Table/AX3_adj_ref_table.csv`
# 
# -----


###############################################################################
#                                   IMPORTS                                   #
###############################################################################

import pandas as pd
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, '/home/Calf Behaviour Classification/PROD/')
 
# importing the paths management file
from PATHS_management import test_import
 
if test_import() :
    print("Import successful")
    from PATHS_management import path_shake_round_info_file, path_raw_ax3_folder, path_time_corr_export_folder, path_adjusted_ax3_folder, path_raw_data_ax3_ref_file
else :
    print("Import failed")


import sys
sys.path.insert(0, '/home/Calf Behaviour Classification/PROD/')
 
# importing the paths management file
from UTILS_data import test_import
 
# Testing the import
if test_import() :
    from UTILS_data import get_data_from_adjusted_AX3_path
    print("Import successful")
else :
    print("Import failed")


# # Drift Correction

# In[3]:


shake_data_available_df = pd.read_csv(path_shake_round_info_file)


# In[4]:


shake_data_available_df.ref_launch_datetime = pd.to_datetime(shake_data_available_df.ref_launch_datetime, format='%Y-%m-%d %H:%M:%S.%f')
shake_data_available_df.ref_stop_datetime = pd.to_datetime(shake_data_available_df.ref_stop_datetime, format='%Y-%m-%d %H:%M:%S.%f')
shake_data_available_df.algo_launch_datetime = pd.to_datetime(shake_data_available_df.algo_launch_datetime, format='%Y-%m-%d %H:%M:%S.%f')
shake_data_available_df.algo_stop_datetime = pd.to_datetime(shake_data_available_df.algo_stop_datetime, format='%Y-%m-%d %H:%M:%S.%f')

shake_data_available_df.head()


# In[5]:


# Identifying the unique calf_ids so it can be used to iterate through the shake rounds. This way the raw data
# per calf needs only to be loaded once 

calf_ids = shake_data_available_df.calf_id.unique()


# In[ ]:


for calf_id in calf_ids:
    print('Processing: ', str(calf_id))
    
    # getting raw data per calf
    raw_data = pd.read_csv(f"{path_raw_ax3_folder}{calf_id}.csv", engine='pyarrow')
    raw_data.DateTime = pd.to_datetime(raw_data.DateTime, format='%Y-%m-%d %H:%M:%S.%f')
    
    # getting shake data per calf
    shake_data = shake_data_available_df[shake_data_available_df.calf_id == calf_id]
    
    for row in shake_data.itertuples():
        # filterign out the raw data based on the shake round launch and stop datetimes
        filtered_raw_data = raw_data[(raw_data.DateTime >= row.algo_launch_datetime) & (raw_data.DateTime <= row.algo_stop_datetime)]
        
        filtered_raw_data = filtered_raw_data.sort_values(['DateTime'])
        filtered_raw_data = filtered_raw_data.reset_index(drop=True)
        
        # step 01 : Synchronising the lauch shake datetime
        launch_gap = (row.ref_launch_datetime - row.algo_launch_datetime).total_seconds()
        
        # if the launch_gap is +, that means for the correction the lauch_gap needs to be added to the algo time
        filtered_raw_data['adjusted_DateTime'] = filtered_raw_data.DateTime + timedelta(seconds = launch_gap)
        
        # step 02: Calculating the drift
        drift = (row.ref_stop_datetime - filtered_raw_data.iloc[-1]['adjusted_DateTime']).total_seconds()
        
        # step 03: drift per entry (correction)
        correction = drift / len(filtered_raw_data)
        
        # Applying the correction
        filtered_raw_data['adjusted_DateTime'] = filtered_raw_data.apply(lambda x: x['adjusted_DateTime'] + timedelta(seconds=correction*x.name), axis=1)
        
        file_name = str(calf_id) + '_' + str(row.ref_launch_datetime) + '_' + str(row.ref_stop_datetime)
        file_name = file_name.replace(' ', '')
        file_name = file_name.replace('-', '')
        file_name = file_name.replace(':', '')
        file_name += "_AX3_adjusted.csv"
        
        #EXPORTS
        filtered_raw_data.to_csv(f"{path_time_corr_export_folder}{file_name}", index=False)
        
print('Complete')


# # Update the referencing table of the adjusted AX3 data

# In[ ]:


adj_file_paths = [f for f in os.listdir(path_adjusted_ax3_folder) if os.path.isfile(os.path.join(path_adjusted_ax3_folder, f))]
print(adj_file_paths)
print(len(adj_file_paths))


# In[ ]:


calf_ids = []
start_dates = []
end_dates = []
file_names = []
for paths in adj_file_paths:
    calf_id, start_date, end_date = get_data_from_adjusted_AX3_path(paths)
    calf_ids.append(calf_id)
    start_dates.append(start_date)
    end_dates.append(end_date) 
    file_names.append(paths.split("/")[-1])


# In[ ]:


df = pd.DataFrame(list(zip(calf_ids, start_dates, end_dates, adj_file_paths, file_names)), columns = ["calf_id", "start_date", "end_date","file_path", "file_name"])
df.sort_values(by=["calf_id", "start_date"], inplace=False)
display(df)


# In[ ]:


df.to_csv(path_raw_data_ax3_ref_file, index=False)


# # Credits
# 
# Author: Oshana Dissanayake 
# 
# Organization: University College Dublin
# 
# Date: April 24, 2023
