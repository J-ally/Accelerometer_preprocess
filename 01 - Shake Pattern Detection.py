#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# This code presents a function to detect a specific shake pattern present inside a raw accelerometer signal. The ideal Shake Pattern signal may look like this :
# 
# ```note
#    (shake)         (shake)         (shake)         (shake)         (shake)        (shake)
#     +++++ __________+++++ __________+++++ __________+++++ __________+++++__________+++++
#             (rest)          (rest)          (rest)          (rest)         (rest)
# ```    
# 
# A Few characteristics :
# - Ideal shake pattern consists of 5 shake and 4 rest instances. Each shake is 5 seconds in duration and each rest   is 10 seconds in duration.
# - But in nature these parameters can be varied. The algorithm presented below here has the capability to handle these variation through the various parameters.
# 
# # Input
# 
# A dataframe that contains the ref times, both for the launch and for the stop of the accelerometer round. The dataframe must contain the following columns (at least) :
# | calf_id | round_trial | launch_ref | stop_ref | 
# |---------|-------------|------------|----------|
# |  1      |      1      | 2022-01-21 11:55:05 | 2022-02-04 09:34:00  |  
# |  ...      |      ...      | ...| ... |   
# 
# # Output
# 
# The output consists of three different dataframes who will be present in the `Export` folder. They can be resumed as follows :
# - The first dataframe contains the ref times for the start and stop of each shake pattern detected. It is named `algo_shake_times_for_reference_shakes.csv`.  
# 
# | calf_id | round_trial | ref_launch_datetime | ref_stop_datetime | algo_launch_datetime  | algo_stop_datetime |
# |---------|-------------|---------------------|-------------------|-----------------------|--------------------|
# |  1      |      1      | 2022-01-21 11:55:05 | 2022-02-04 09:34:00  | 2022-01-21 11:55:06 | 2022-02-04 09:34:01 |
# |  ...    |      ...    | ...| ... | ... | ... |
# 
# - The second dataframe contains the rows of the preceeding dataframes who have both a valid launch and stop time, the file is named : `algo_data_available_for_complete_shake_rounds.csv`.
# 
# - The last dataframe contains the rows of the first dataframes where the difference between the ref launch and algo launch times is more than 60 seconds, as well as between the ref stop and algo stop times. Those times will be considered unusual and need a manual inspection : `manual_inspection_needed.csv`.
# 
# # Paths
# Documentation of the paths in the `PATHS_management.py` file at the root of the `PROD` folder.
# ------


###############################################################################
#                                 IMPORTS                                     #
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import glob
import statistics
from datetime import timedelta, datetime
import numpy as np
from datetime import timedelta
import time
import pickle
from scipy import stats

plt.rcParams['figure.figsize'] = (30,8)


###############################################################################
#                              PATH MANAGEMENT                                #
###############################################################################

# importing the paths management file
from PATHS_management import test_import

# Testing the import
if test_import() :
    print("Import of the paths successful")
    from PATHS_management import *
else :
    print("Import of the paths failed")


###############################################################################
#                                 FUNCTIONS                                   #
###############################################################################


def validate_rest_durations(rest_arr, minimum_rest_duration, maximum_rest_duration):
    return all(minimum_rest_duration <= x <= maximum_rest_duration for x in rest_arr)


def perform_moving_average(signal, average_out_window_len, average_out_window_threshold):
    averaged_out_signal = []
    for i in range(len(signal)):
        start = max(0, i - average_out_window_len // 2)
        end = min(len(signal), i + average_out_window_len // 2 + 1)
        window = signal[start:end]

        av = statistics.mean(window)
        if av >= average_out_window_threshold:
            averaged_out_signal.append(1)
        else:
            averaged_out_signal.append(0)
            
    return averaged_out_signal


def detect_bit_change(window):
    # Create empty lists to store indexes
    zeros_to_ones = []
    ones_to_zeros = []

    # Loop through the array and check for changes
    for i in range(1, len(window)):
        if window[i] == 1 and window[i-1] == 0:
            zeros_to_ones.append(i)
        elif window[i] == 0 and window[i-1] == 1:
            ones_to_zeros.append(i)
            
    return [ones_to_zeros, zeros_to_ones]


def trim_zeros(arr):
    start = 0
    end = len(arr)

    # Find the first non-zero element
    while start < end and arr[start] == 0:
        start += 1

    # Find the last non-zero element
    while end > start and arr[end-1] == 0:
        end -= 1

    return arr[start:end]


def find_first_and_last_shake_indexes(signal):
    # find first index
    first_index = -1
    for i in range(len(signal)):
        if(i+5 >= len(signal)):
            break
        if statistics.mode(signal[i:i+5]) == 1:
            if((signal[i] == 0 and signal[i+1] == 1) or (signal[i] == 1 and signal[i+1] == 1) or (signal[i] == 1 and signal[i+1] == 0)):
                first_index = i
                break
            else:
                i+=1
                while statistics.mode(signal[i:i+5]) == 1:
                    if((signal[i] == 0 and signal[i+1] == 1) or (signal[i] == 1 and signal[i+1] == 1) or (signal[i] == 1 and signal[i+1] == 0)):
                        first_index = i
                        break
                    i+=1
            break
            
    # find last index
    last_index = -1
    for j in range(len(signal)-1, 0, -1):
        if(j-5 < 0):
            break
        if statistics.mode(signal[j-4:j+1]) == 1:
            if((signal[j] == 0 and signal[j-1] == 1) or (signal[j] == 1 and signal[j-1] == 1) or (signal[j] == 1 and signal[j-1] == 0)):
                last_index = j
                break
            else:
                j-=1
                while statistics.mode(signal[j-4:j+1]) == 1:
                    if((signal[j] == 0 and signal[j-1] == 1) or (signal[j] == 1 and signal[j-1] == 1) or (signal[j] == 1 and signal[j-1] == 0)):
                        last_index = j
                        break
                    j-=1
            break
    
    return first_index, last_index


def calculate_shake_rest_durations(shake_rest_indexes, len_of_cleaned_signal):
    
    shake_durations = []
    rest_durations = []
    
    for i in range(len(shake_rest_indexes[0])):
        rest_durations.append(shake_rest_indexes[1][i] - shake_rest_indexes[0][i])

    for i in range(-1,len(rest_durations)):
        if(i == -1):
            shake_durations.append(shake_rest_indexes[0][0])
        elif(i == (len(rest_durations)-1)):
            shake_durations.append(len_of_cleaned_signal - shake_rest_indexes[1][i])
        else:
            shake_durations.append(shake_rest_indexes[0][i+1] - shake_rest_indexes[1][i])
            
    return [shake_durations, rest_durations]


def signal_preprocess(signal, shake_threshold, minimum_shake_count, minimum_rest_count, noise_cleaning_function,
                      *noise_cleaning_function_args):
    
    signal_binary = [0 if x < shake_threshold else 1 for x in signal]

    start,end = find_first_and_last_shake_indexes(signal_binary)
    
    if(start == -1 or end == -1):
        return [0, None, None]

    # atleast 1 bit should be there to represent each shake and rest period after trimming
    if len(signal_binary[start:end]) <= (minimum_shake_count + minimum_rest_count):
        return [0, None, None]
    else:
        # averaging out the signal using moving average
        trimmed_signal = trim_zeros(signal_binary[start:end])
        cleaned_out_signal = noise_cleaning_function(trimmed_signal, *noise_cleaning_function_args)
        return [1, cleaned_out_signal, [start,end]]
    
    
def shake_pattern_detector(signal, shake_threshold = 0.75, 
                           minimum_rest_count=4, maximum_rest_count=5,
                           minimum_shake_count=5, maximum_shake_count=6, 
                           minimum_rest_duration=8, maximum_rest_duration=13, 
                           minimum_shake_duration=3, maximum_shake_duration=7,
                           average_out_window_len=3, average_out_window_threshold=0.5):
    
    noise_cleaning_args = [average_out_window_len, average_out_window_threshold]
    
    processed_signal = signal_preprocess(signal, shake_threshold, minimum_shake_count, minimum_rest_count, 
                                        perform_moving_average, *noise_cleaning_args)
    
    if(processed_signal[0] == 1):
        # return shake indexes first
        shake_rest_indexes = detect_bit_change(processed_signal[1])
        
        if(len(shake_rest_indexes[1]) >= minimum_rest_count):
            # return shake durations first
            shake_durations, rest_durations = calculate_shake_rest_durations(shake_rest_indexes, 
                                                                             len(processed_signal[1]))
            if((minimum_shake_count <= len(shake_durations) <= maximum_shake_count) and 
               (minimum_rest_count <= len(rest_durations) <= maximum_rest_count) and 
               validate_rest_durations(rest_durations, minimum_rest_duration, maximum_rest_duration)):
                return[1, [processed_signal[2][0],processed_signal[2][1]], shake_durations[0] + shake_durations[-1]]
            else:
                return[0]
        else:
            return[0]
    else:
        return[0]
    
    
# flag = 0 for launch shake and flag = 1 for end shake
def detect_shake_patterns_in_rounds(raw_data, flag, min_time_consider, max_time_consider):
    
    if(flag == 0):
        start_time = row.launch_ref - timedelta(minutes = min_time_consider)
        end_time = row.launch_ref + timedelta(minutes = max_time_consider)
    elif(flag == 1):
        start_time = row.stop_ref - timedelta(minutes = min_time_consider)
        end_time = row.stop_ref + timedelta(minutes = max_time_consider)
        
    analyze_data = raw_data[(raw_data.DateTime >= start_time) & (raw_data.DateTime <= end_time)]
        
    if(len(analyze_data) > 0):

        det_shake_pattern_datetime = []
        det_shake_pattern_first_n_last_shake_len_sum = []

        for i in range(len(analyze_data)):
            sub_start_time = start_time + timedelta(seconds=i)
            sub_end_time = sub_start_time + timedelta(seconds=85)

            sub_analyze_data = analyze_data[(analyze_data.DateTime >= sub_start_time) & 
                                            (analyze_data.DateTime <= sub_end_time)]

            res = shake_pattern_detector(sub_analyze_data.Amag, shake_threshold=0.75)

            if(res[0] == 1):
                det_shake_pattern_datetime.append(list(sub_analyze_data.DateTime)[res[1][0]+1])
                det_shake_pattern_first_n_last_shake_len_sum.append(res[2])
            else:
                end_result = 'No Pattern Detected'

        if(len(det_shake_pattern_datetime) > 0):
            # maximum first shake len gives the final pattern
            max_shake_len_index = np.argmax(det_shake_pattern_first_n_last_shake_len_sum)
            return det_shake_pattern_datetime[max_shake_len_index]
        else:
            return 'No Pattern Detected'
    else:
        return 'No Data Available'


if __name__ == "__main__":
    #### Detecting Reference Shake Patterns
    
    # Import reference times
    ref_times = pd.read_csv(path_reference_shake_data_file, sep=";", parse_dates=['launch_ref', 'stop_ref'])
    print(ref_times.head())
    
    """
    ref_times.launch_ref = pd.to_datetime(ref_times.launch_ref)
    ref_times.stop_ref = pd.to_datetime(ref_times.stop_ref)

    ref_times.head()

# In[6]:


ref_times = ref_times[['calf_id', 'round_trial', 'launch_ref', 'stop_ref']]
ref_times = ref_times.reset_index(drop=True)


# In[7]:


print('Total Number of Shake Times to detect: ', len(ref_times))
print('Total Number of Rows with null/empty values: ', ref_times.isna().any(axis=1).sum())


# In[6]:


# filtering out the calves based on the raw data availability 
calf_ids = ref_times.calf_id.unique()

raw_data_csvs = glob.glob(path_raw_ax3_folder + '*.csv')

raw_data_available_calf_ids = []
for csv in raw_data_csvs:
    raw_data_available_calf_ids.append(int(csv.split('/')[-1].split('.')[0]))
    
filtered_calf_ids = []
for ref_calf_id in calf_ids:
    if(ref_calf_id not in raw_data_available_calf_ids):
        print('Raw data not available for calf_id: ', ref_calf_id)
    else:
        filtered_calf_ids.append(ref_calf_id)


# In[12]:


master_data = []

fn_start_time = time.time()

for calf_id in filtered_calf_ids:
    print('Processing: ', calf_id)
    ref_data_filtered = ref_times[ref_times.calf_id == calf_id]
    
    raw_data = pd.read_csv(f"{path_raw_ax3_folder}{calf_id}.csv", usecols=['DateTime', 'Amag'], engine='pyarrow')
    raw_data.DateTime = pd.to_datetime(raw_data.DateTime, format='%Y-%m-%d %H:%M:%S.%f')
    
    raw_data = raw_data.set_index('DateTime')
    raw_data = raw_data.resample('1S').mean()
    raw_data['DateTime'] = raw_data.index.copy()
    
    for row in ref_data_filtered.itertuples():
        
        # ----------------- Launch Shake -----------------
        launch_result = detect_shake_patterns_in_rounds(raw_data, 0, 90, 90)
            
        # ----------------- End Shake -----------------
        end_result = detect_shake_patterns_in_rounds(raw_data, 1, 90, 90)

        master_data.append([row.calf_id, row.round_trial, row.launch_ref, row.stop_ref, 
                           launch_result, end_result])
        
    del raw_data

fn_end_time = time.time()
execution_time = (fn_end_time - fn_start_time)


# In[13]:


execution_time_mins = execution_time/60
print('Execution time: ', execution_time_mins)


# In[14]:


master_df = pd.DataFrame(master_data, columns = ['calf_id', 'round_trial', 'ref_launch_datetime', 'ref_stop_datetime', 
                                                 'algo_launch_datetime', 'algo_stop_datetime'])


# # Testing

# In[1]:


len_valid_datetimes = 0

for row in master_df.itertuples():
    try:
        pd.to_datetime(row.algo_launch_datetime)
        len_valid_datetimes+=1
    except:
        continue
        
for row in master_df.itertuples():
    try:
        pd.to_datetime(row.algo_stop_datetime)
        len_valid_datetimes+=1
    except:
        continue

print('AX3 Time available shakes (Without considering as rounds): ',len_valid_datetimes)


# In[16]:


master_df_launch_no_data_available = master_df[master_df.algo_launch_datetime == 'No Data Available']
master_df_end_no_data_available = master_df[master_df.algo_stop_datetime == 'No Data Available']
print('No data available: ', len(master_df_launch_no_data_available) + len(master_df_end_no_data_available))

master_df_launch_no_pattern_detected = master_df[master_df.algo_launch_datetime == 'No Pattern Detected']
master_df_end_no_pattern_detected = master_df[master_df.algo_stop_datetime == 'No Pattern Detected']
print('No pattern detected: ', len(master_df_launch_no_pattern_detected) + len(master_df_end_no_pattern_detected))


# In[ ]:


manual_inspection = []
for row in master_df.itertuples():
    if datetime.timedelta(row.algo_launch_datetime - row.ref_launch_datetime).total_seconds() > 60 \
        or datetime.timedelta(row.algo_launch_datetime - row.ref_launch_datetime).total_seconds() < -60:

        manual_inspection.append([row.calf_id, row.round_trial, row.ref_launch_datetime, row.ref_stop_datetime, 
                           row.algo_launch_datetime, row.algo_stop_datetime])
        
    elif datetime.timedelta(row.algo_stop_datetime - row.ref_stop_datetime).total_seconds() > 60 \
        or datetime.timedelta(row.algo_stop_datetime - row.ref_stop_datetime).total_seconds() < -60:

        manual_inspection.append([row.calf_id, row.round_trial, row.ref_launch_datetime, row.ref_stop_datetime, 
                           row.algo_launch_datetime, row.algo_stop_datetime])

manual_inspection_df = pd.DataFrame(manual_inspection, columns = ['calf_id', 'round_trial', 'ref_launch_datetime', 'ref_stop_datetime',
                                                                  'algo_launch_datetime', 'algo_stop_datetime'])
display(manual_inspection_df)


# ### Identifying shake rounds where both the ref_launch and ref_stop has been detected by the algorithm (so a complete round can be considered)

# In[18]:


shake_data_available = []

for row in master_df.itertuples():
    try:
        pd.to_datetime(row.algo_launch_datetime)
        try:
            pd.to_datetime(row.algo_stop_datetime)
            shake_data_available.append(row)
        except:
            pass
    except:
        continue
        
shake_data_available_df = pd.DataFrame(shake_data_available)
shake_data_available_df = shake_data_available_df.drop('Index', axis=1)
shake_data_available_df = shake_data_available_df.sort_values(['calf_id', 'ref_launch_datetime'])
shake_data_available_df = shake_data_available_df.reset_index(drop=True)
print(len(shake_data_available_df))


# # Exports

# In[17]:


master_df.to_csv(f"{path_SPD_export_folder}algo_shake_times_for_reference_shakes.csv", index=False)
manual_inspection_df.to_csv(f"{path_SPD_export_folder}manual_inspection_needed.csv", index=False)
shake_data_available_df.to_csv(f"{path_SPD_export_folder}algo_data_available_for_complete_shake_rounds.csv", index=False)


# # Credits
# 
# Authors: Oshana Dissanayake
# 
# Organization: University College Dublin
# 
# Date: April 24, 2023
"""