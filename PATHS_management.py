# -*- coding: utf-8 -*-
"""
Created on June 12 2023
@author:  Lucile, Oshana, Joseph
"""

###############################################################################
#                               GUIDELINES                                    #
###############################################################################
#
# This file contains all the common paths used for the project.
# It is used in all the other files of PROD and allow a streamlined way of moving
# files if needed, without changing all the paths individually in the code.
# All the paths are absolute paths, and should be changed if the project is moved
#
##### NAMING CONVENTIONS
# - All the paths are in lower case
# - All the paths must be named in this format : 
#               
#               path_WHATEVER_DESCRIPTIVE_NAME_file/folder
 
#   depending on the case (file when it is a file, folder when it is a folder)
#   In the case of a FOLDER, the path must end with a /
#   In the case of a FILE, the path must end with the file extension (.csv, .py, .txt, etc.)


###############################################################################
#                               IMPORT TESTING                                #
###############################################################################
# The following function can be used to test if the file is imported correctly:

def test_import () -> bool:
    """
    Returns True if the file i
    s imported correctly.
    Returns:
        True (bool): True if the file is imported correctly.
    """
    print("PATHS_management.py imported successfully")
    return True

###############################################################################
#                              IMPLEMENTATION                                 #
###############################################################################

# To import the paths, USE THE FOLLOWING CODE :

# import sys
# sys.path.insert(0, '/home/Calf Behaviour Classification/PROD/')
 
# # importing the paths management file
# from PATHS_management import test_import
 
# # Testing the import
# if test_import() :
#     from PATHS_management import *
# else :
#     print("Import failed")

###############################################################################
#                               COMMON PATHS                                  #
###############################################################################

# Path to the folder containing the raw data
path_raw_ax3_folder = '/home/Work/Dehorning_Analysis/DataGroups/CleanedData/'
# Path to the folder containing all the labelling files (manual annotations from the annotation software)
path_parent_all_labelling_folder = "/home/admin/Calf_Video_labelling/Labelling_Files/"
# Path to the file containing the reference times for the shake patterns
path_reference_shake_data_file = '/home/Calf Behaviour Classification/PROD/Files/20230327_date_ref_pattern.csv'
# Path to the folder containing the adjusted raw accelerometer data 
path_adjusted_ax3_folder = "/home/Calf Behaviour Classification/PROD/Files/AX3_adjusted_per_shake_round/"
# Path to the folder containing the processed label files
path_processed_labels_folder = "/home/Calf Behaviour Classification/PROD/Files/Boris_processed_label_files/"
# Path to the folder containing the label files that shouldn't be processed
path_boris_to_not_process = "/home/Calf Behaviour Classification/PROD/Files/Boris_processed_label_files/To check/"
# Path to the folder containing the annotated AX3 data with the labels
path_adjusted_ax3_with_labels_folder =  "/home/Calf Behaviour Classification/PROD/Files/AX3_adjusted_with_labels/"
# Path to the reference table for the AX3 data containing the reference times for the shake patterns
path_raw_data_ax3_ref_file = "/home/Calf Behaviour Classification/PROD/Files/AX3_adjusted_per_shake_round/Reference Table/AX3_adj_ref_table.csv"

# Path to the final dataset by rows
path_dataset_final_by_rows_file = "/home/Calf Behaviour Classification/PROD/Files/dataset/whole data/2023-06-27_FINAL_dataset_by_rows.csv"
path_dataset_final = "/home/Calf Behaviour Classification/PROD/Files/datasets/whole data/2023-08-10_FULL_dataset.csv"

###############################################################################
#                              PATH MANAGEMENT                                #
###############################################################################

# for step 1 in prod
path_SPD_export_folder = '/home/Calf Behaviour Classification/PROD/Codes/01 - Shake Pattern Detection/Exports/'

# for step 2 in prod
path_shake_round_info_file = '/home/Calf Behaviour Classification/PROD/Codes/01 - Shake Pattern Detection/Exports/algo_data_available_for_complete_shake_rounds.csv'
path_time_corr_export_folder = '/home/Calf Behaviour Classification/PROD/Codes/02 - Time Drift Correction/Exports/'

# for step 3 in prod
path_output_adj_annotations_folder = "/home/admin/Calf_Video_labelling/Labelling Files gestion/"
path_output_boris_adjusted_folder = "/home/admin/Calf_Video_labelling/Labelling Files gestion/Raw Boris Adjusted Times/"
path_output_figs_ax3_folder = "/home/admin/Calf_Video_labelling/Labelling Files gestion/Labels Figures Testing Output/"
path_output_pkl_ax3_folder = "/home/admin/Calf_Video_labelling/Labelling Files gestion/Labels Figures Testing Output/Pickle Files/"

# Path to the dataset folder
path_dataset_final_folder = "/home/Calf Behaviour Classification/PROD/Files/dataset/whole data/"

path_available_data_file = "/home/Calf Behaviour Classification/PROD/Files/Video synchro/cleaned_camera_n_reference_times.csv"
path_already_adjusted_files_file = "/home/admin/Calf_Video_labelling/Labelling Files gestion/df_adjusting_data_tracking.csv"
path_already_adjusted_plots_file = "/home/admin/Calf_Video_labelling/Labelling Files gestion/df_adjusting_plot_tracking.csv"

path_manual_inspection_output_file = "/home/Calf Behaviour Classification/PROD/Codes/03 - Label Time Correction/AX3_Camera_sync_manual_inspection_outputs.csv"


# for step 4 in prod
path_shake_rounds_info_file = '/home/calf_stress_detection/oshana_2023/Camera Time Synchronisation/Camera Time Synchronisation Code V3/Derived Metadata/shake_rounds_with_videos.csv'
path_general_ethogram_file = '/home/Calf Behaviour Classification/PROD/Codes/04 - AX3 and labels alignment/general_ethogram.csv'

path_export_long_dataset_folder = "/home/Calf Behaviour Classification/PROD/Files/dataset/dataset/"
path_export_train_long_dataset_folder = "/home/Calf Behaviour Classification/PROD/Files/datasets/dataset/train/"
path_export_test_long_dataset_folder = "/home/Calf Behaviour Classification/PROD/Files/datasets/dataset/test/"
path_export_val_long_dataset_folder = "/home/Calf Behaviour Classification/PROD/Files/datasets/dataset/validation/"


# path_export_short_dataset_folder = "/home/Calf Behaviour Classification/PROD/Files/dataset/SHORT_dataset/"
# path_export_train_short_dataset_folder = "/home/Calf Behaviour Classification/PROD/Files/dataset/SHORT_dataset/train/"
# path_export_test_short_dataset_folder = "/home/Calf Behaviour Classification/PROD/Files/dataset/SHORT_dataset/test/"
# path_export_val_short_dataset_folder = "/home/Calf Behaviour Classification/PROD/Files/dataset/SHORT_dataset/validation/"


# for step 5 in prod

# Path to the file containing the files that have been used to construct the dataset
path_adjusted_files_indexed_dataset_file = "/home/Calf Behaviour Classification/PROD/Files/dataset/adjusted_files_indexed.txt"
# Path to the file containing the files that have NOT been used to construct the dataset (no adjusted AX3 data found)
path_adjusted_files_NOT_indexed_dataset_file = "/home/Calf Behaviour Classification/PROD/Files/dataset/adjusted_files_couldnt_index.txt"

