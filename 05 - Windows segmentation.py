#!/usr/bin/env python
# coding: utf-8

# # Documentation
# 
# This scripts aims to segment a dataset into windows of a particular time.
# 
# This time can be set below, in the wariable `window_size`.
# 
# This code outputs a dataset named 

# In[1]:


import pandas as pd
import numpy as np
import pycatch22 as catch22 
from datetime import datetime, timedelta
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


# # Importing the features script

# In[2]:


import sys
sys.path.insert(0, '/home/Calf Behaviour Classification/PROD/')
 
# importing the paths management file
from CLASSIF_UTILS import test_import
 
# Testing the import
if test_import() :
    from CLASSIF_UTILS import apply_data_augmentation, Features_Calculations, get_block_proportions  # or from UTILS_data import *specfic function name*
    print("Import successful")
else :
    print("Import failed")


# In[10]:


##### Calculate the feature and create the dataset at the same time

##### CHANGE THE PATH IF NEEDED (need to be the dataset to explore)

df_whole_data = pd.read_csv(path_dataset_final, sep = ',', engine='pyarrow')
df_whole_data["adjusted_DateTime"] = pd.to_datetime(df_whole_data["adjusted_DateTime"])
display(df_whole_data.head())


# # Defining the features

# In[11]:


# the feature names
feature_names = []

text_file = open("/home/Work/Joseph/Distance-based algo/Ranfom Forest/features_name.txt", "r")
lines = text_file.readlines()
for line in lines:
    feature_names.append(line.strip())

print(lines)

print(feature_names)

basic_columns = ["calf_id", "label"]

feature_table = pd.DataFrame(columns = basic_columns)
display(feature_table)


# # Looking for the best Window Size

# In[12]:


all_labels = df_whole_data["label"].unique()


time_beha = {label : [] for label in all_labels}
med_label = {label : [] for label in all_labels}
min_label = {label : [] for label in all_labels}
max_label = {label : [] for label in all_labels}

for label in all_labels :
    df_label = df_whole_data[df_whole_data["label"] == label]
    block_ids= list(df_label["block_id"].unique())
    
    for block in block_ids :
        # print(block["start_time"])
        block = df_label[df_label["block_id"] == block]
        time_passed =  block.iloc[-1]["adjusted_DateTime"] - block.iloc[0]["adjusted_DateTime"]

        print(time_passed)
        if time_passed < pd.Timedelta("0s") :
            print("negative time passed")
            display(block)
        time_beha[label].append(time_passed.total_seconds())


        
med_label = {label : np.median(time_beha[label]) for label in all_labels}
min_label = {label : np.min(time_beha[label]) for label in all_labels}
max_label = {label : np.max(time_beha[label]) for label in all_labels}


print("median of the lenghts of the labels : ",med_label)
print("min of the lenghts of the labels ",min_label)
print("max of the lenghts of the labels ",max_label)

print("\n\n====THE FINAL SIZES OF THE WINDOWS (minimum of the medians)")
print("the size of the window choosen : ", np.min(list(med_label.values())))


# # Defining the parameters

# In[13]:


window_size = 3 # (in seconds)

#get the labels
all_labels = df_whole_data["label"].unique()

path_name_output = f"/home/Calf Behaviour Classification/PROD/Codes/06 - Window segmentation/Output/DETAILED_{window_size}s_feature_table.csv"


# # Running the script

# In[15]:


for label in all_labels :
    df_label = df_whole_data[df_whole_data["label"] == label]
    blocks = list(df_label["block_id"].unique())
    print(blocks)
    
    for block in tqdm(blocks, desc=f"Calculating features for blocks of {window_size}s, Label : {label}") :
        block = df_label[df_label["block_id"] == block]
        
        total_time_block = block.iloc[-1]["adjusted_DateTime"] - block.iloc[0]["adjusted_DateTime"]

        if total_time_block < timedelta(seconds = window_size) :
            # the block is too small to be considered
            continue
        
        # possible blocks to calculate the features on
        nb_possible_sub_blocks = int(total_time_block.seconds / window_size)
        
        for i in range(nb_possible_sub_blocks-1) :
            # create a sub-block
            time_start = block.iloc[0]["adjusted_DateTime"] + timedelta(seconds = i * window_size)
            time_end = block.iloc[0]["adjusted_DateTime"] + timedelta(seconds = (i+1) * window_size)
            sub_block = block[(block["adjusted_DateTime"] >= time_start) & (block["adjusted_DateTime"] < time_end)]
            
            # create a temporary dataframe to store the features of the sub-block
            temp_df = pd.DataFrame(columns = basic_columns)
            temp_df["calf_id"] = sub_block["calf_id"]
            temp_df["label"] = label
            temp_df["Accx"] = sub_block["Accx"]
            temp_df["Accy"] = sub_block["Accy"]
            temp_df["Accz"] = sub_block["Accz"]
            temp_df["Amag"] = sub_block["Amag"]
            temp_df["adjusted_DateTime"] = sub_block["adjusted_DateTime"]
            
            # calculate all the features in features_names for the sub-block
            features = Features_Calculations(temp_df, feature_names, False)
            dict_final = features.get_feature_dict()
            
            # if len(dict_final) != len(feature_names) :
            #     raise Exception("The number of features calculated is not the same as the number of features expected")
            
            # display(temp_df.head())
            # print(dict_final)
            
            feature_table = pd.concat([feature_table, pd.DataFrame(dict_final, index = [0])], ignore_index = True)
            # feature_table.iloc[-1, 0] = sub_block["calf_id"].values[0]
            # feature_table.iloc[-1, 1] = label
            # display(feature_table.head())
            
            
            # UNCOMMENT THOSE (BREAK) TO RUN ON THE WHOLE DATASET
    #         break
    #     break
    # break


# In[13]:


display(feature_table.head())
print(feature_table.shape)


# In[14]:


feature_table.dropna(inplace = True)
print(feature_table.shape)


# In[16]:


print(feature_table["label"].value_counts(normalize = False))


# In[ ]:


# Save the feature table
feature_table.to_csv(path_name_output, index = False)

