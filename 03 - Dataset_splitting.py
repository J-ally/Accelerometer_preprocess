# -*- coding: utf-8 -*-
"""
Created on August 11 2023
@author: Joseph
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import time
import argparse
from tqdm import tqdm
from itertools import combinations
from datetime import datetime, date, timedelta


class Dataset_split :
    """
    Class to split the dataset into train test and validation sets
    
    The new sets will respect the following proportions :
        - train : 70% of the dataset
        - test : 15% of the dataset
        - validation : 15% of the dataset
    
    Each dataset will have different calves (no overcrossing), but the same proportion of each behaviour
    (or at least as close as possible).
    
    Full dataset is a dataframe containing all the data, with the following columns :
        - label : the behaviour of the calf
        - calf_id : the name of the calf
    """
    
    def __init__ (self, full_dataset : pd.DataFrame, grouping_dict : dict, path : str, \
                train_prop : float = 0.7, test_prop : float = 0.15, validation_prop : float = 0.15, \
                verbose : bool = True) :
        """
        Constructor of the class

        Args:
            full_dataset (pd.DataFrame) : the dataframe containing all the data
            grouping_dict (dict) : the dict of the groupings of the behaviours
            train_prop (float, optional) : the proportion of the full dataset to be allocated to the training set (Defaults to 0.7)
            test_prop (float, optional) : the proportion of the full dataset to be allocated to the testing set (Defaults to 0.15)
            val_prop (float, optional) : the proportion of the full dataset to be allocated to the validation set (Defaults to 0.15)
        """
        
        self.dataset = full_dataset
        self.path_output = path
        self.grouping_dict = grouping_dict
        self.verbose = verbose  
        self.train_prop = train_prop
        self.test_prop = test_prop
        self.val_prop = validation_prop
                
        self.train_set = pd.DataFrame()
        self.test_set = pd.DataFrame()
        self.validation_set = pd.DataFrame()
        
        ####### PIPELINE IS HAPPENING HERE #######
        
        # regrouping the behaviours
        self.regrouped_dataset = self.regroup_behaviours()
        
        self.create_summary_table()
        
        #get the original proportions of each behaviour in the dataset
        self.regrouped_proportions = self.regrouped_dataset["label"].value_counts(normalize = True)*100
        if self.verbose : 
            print("The proportions of the regrouped dataset : \n",self.regrouped_proportions)
        
        # get the behaviours for each calf in the dataset
        self.calf_with_full_behaviours = self.get_all_beha_calves()
        # print("calves with all the behaviours : ", self.calf_with_full_behaviours)
        
        self.calf_proportions = self.regrouped_dataset["calf_id"].value_counts(normalize = True)*100
        # print("The total calf representation proportion (%) : \n", self.calf_proportions)
        
        # get the number of calves to be allocated to the test and validation sets
        self.selected_test_val_calves = self.select_the_test_and_train_calves()
        
        # finally create and save the datasets
        self.save_the_datasets()
        
        
    def regroup_behaviours (self) :
        """
        Regroup the behaviours in the dataset according to the self.grouping_dict
        Args:
            behavior_grouping (dict): the dict of the groupings of the behaviours
                Need to have the followwing format :
                    behavior_groups = {'lying' :['lying-down', 'lying'],
                                       'standing': ["rising", "standing", "SRS"], ...}
        Returns:
            Changes the dataset in place by modifying the "label" column
        """
        dataset = self.dataset.copy()
        
        # first check if all the behaviours in the dataset are in the grouping dict
        keys_list = list(self.grouping_dict.keys())
        values = list(self.grouping_dict.values())
        values_list = list(np.concatenate(values).flat) 

        row_to_drop = []
        
        # Then grouping the behaviours
        print("\n--- Regrouping the behaviours\n")
        for row in tqdm(dataset.itertuples(), total=dataset.shape[0]) :
            if row.label in values_list :
                # in place change for better performance of the predictions
                dataset.at[row.Index, "label"] = [key for key, value in self.grouping_dict.items() if row.label in value][0]
                continue
            
            else :
                # the behaviour is not in the grouping dict            
                # print(f"row {row.Index} : {row.label} not in the grouping dict")
                row_to_drop.append(row.Index)

        # drop the rows that are not in the grouping dict
        dataset.drop(row_to_drop, inplace = True)
        dropped_dataset = self.dataset.loc[row_to_drop]
        
        if self.verbose :
            all_beha = list(dataset["label"].unique())
            
            print("\nnew behaviours : " , all_beha)   
            # print("row dropped : ", row_to_drop)
            print("len row dropped unique : ", len(row_to_drop))
            print("proportion of rows dropped (%) : ", len(row_to_drop)/dataset.shape[0]*100)
            print("the most dropped behaviour : \n", dropped_dataset["label"].value_counts()[:5], "\n")
        
        print("\n--- Behaviours regrouped ! ---")
        
        return dataset
    
    def create_summary_table (self) :
        """
        Create a summary table of the dataset
        Saves the summary table in the self.path_output folder
        """
        
        df_regrouped_summary = self.regrouped_dataset["label"].value_counts().rename_axis('labels').reset_index(name='counts').sort_values(by = "labels", ascending=True)
        df_regrouped_summary["counts_proportion"] = df_regrouped_summary["counts"] / df_regrouped_summary["counts"].sum()
        df_regrouped_summary["nb_calf"] = self.regrouped_dataset.groupby("label")["calf_id"].nunique().values
        df_regrouped_summary["calf_ids"] = list(self.regrouped_dataset.groupby("label")["calf_id"].unique())
        df_regrouped_summary["time (min)"] = round(df_regrouped_summary["counts"]/(8*60),1)
        df_regrouped_summary["time (hour)"] = round(df_regrouped_summary["counts"]/(8*60*60),1)
       
        list_total = ["TOTAL",sum(list(df_regrouped_summary["counts"])),sum(list(df_regrouped_summary["counts_proportion"])), " ", " ", sum(list(df_regrouped_summary["time (min)"])), sum(list(df_regrouped_summary["time (hour)"]))]
        df_regrouped_summary.loc[len(df_regrouped_summary)] = list_total
        
        # print(df_regrouped_summary)
        
        print(f"\nThe total number of calves in the dataset : {len(self.regrouped_dataset['calf_id'].unique())}")

        print(f"\nThe total time of annotations (hour) : {df_regrouped_summary.iloc[-1]['time (hour)']}")
        
        if len(self.grouping_dict.keys()) >= 6 :
            
            df_train.to_csv(f"{self.path_output}/train/{date.today()}_LONG_DETAILED_train_{self.train_prop}_dataset.csv", index=False)
            
        if len(self.grouping_dict.keys()) < 6 :
            df_train.to_csv(f"{path}/train/{date.today()}_LONG_SHORT_train_{self.train_prop}_dataset.csv", index=False)
        print("\n--- The datasets have been saved ---")
    
    

    
    def get_all_beha_calves (self) :
        """
        Get the list of all the calves having all the behaviours in the dataset
        """
        
        print("\n--- Getting the calves having all the behaviours in the dataset\n")
    
        # the number of calves having all the behaviours
        grouped_calf_ids = self.regrouped_dataset["calf_id"].unique()
        grouped_max_beha = len(self.grouping_dict.keys())
        if self.verbose :
            print(f"Number of calves in the dataset : {len(grouped_calf_ids)}")
            print(f"Number of behaviours in the dataset : {grouped_max_beha}")

        self.behaviours_per_calf = {}
        calf_all_beha = []

        for calf_id in grouped_calf_ids :
            nb_behaviours_by_calf = len(self.regrouped_dataset[self.regrouped_dataset["calf_id"] == calf_id]["label"].unique())
            behaviours_by_calf = self.regrouped_dataset[self.regrouped_dataset["calf_id"] == calf_id]["label"].unique()
            self.behaviours_per_calf[calf_id] = behaviours_by_calf
        
            if nb_behaviours_by_calf == grouped_max_beha :
                calf_all_beha.append(calf_id)
            
            # if self.verbose :
            #     print(f"{calf_id}")
            #     print(f"Number of behaviours for calf {calf_id} : {nb_behaviours_by_calf}")
            #     print(f"Behaviours for calf {calf_id} : {behaviours_by_calf}")
            
        
        if self.verbose :
            print(f"\nNumber of calves having all the behaviours : {len(calf_all_beha)}")
            print(f"Calves having all the behaviours : {calf_all_beha}")
        
        print("\n--- All the calves having all the behaviours have been retrieved ! ---\n")
        
        return calf_all_beha
    
    
    def select_the_test_and_train_calves (self) :
        """
        Select the calves for the train and test sets (need to have all the behaviours)
        
        Returns:
            (test_combination, val_combination) : the best combination of calves for the test and validation sets which mininize the difference between the train and test sets
            and have different calves for the test and validation sets and train set without overlap
        """
        print("\n--- Selecting the calves for the train and test sets")
        
        full_calf_proportions = self.calf_proportions[self.calf_with_full_behaviours]
        full_calf_proportions_sum = full_calf_proportions.sum()
        if self.verbose :
            print("\nThe calf who have all the behaviour representation proportion (%): \n", full_calf_proportions )
            print("\nThe total representation of the test and validation with those calves (%) : ", full_calf_proportions_sum)
        
        # get the calves to be allocated to the test sets
        working_test_combinations = []
        for nb_calf in range(len(self.calf_with_full_behaviours), 0, -1) :
            # try all the combinations of calves to get to the right distribution
            calf_combinations = list(combinations(self.calf_with_full_behaviours, nb_calf))
            for calf_combination in calf_combinations :
                total_proportion = sum([full_calf_proportions[calf] for calf in calf_combination])
                total_proportion = round(total_proportion, 0)
                # print(f"nb calf {nb_calf}, combination {calf_combination}, sum proportion : {total_proportion}")
                # chek if we get to the right distribution
                if total_proportion == self.test_prop*100 or total_proportion == (self.test_prop*100-1) or total_proportion == (self.test_prop*100+1) :
                    working_test_combinations.append(calf_combination)
                    if self.verbose :
                        print(f"---Test working combination {calf_combination}, sum proportion : {total_proportion}")
        
        if working_test_combinations == [] :
            raise ValueError("No combination of calves can be found to get to the right distribution for the test set")
        
        # get the calves to be allocated to the test sets
        working_val_combinations = []
        for nb_calf in range(len(self.calf_with_full_behaviours), 0, -1) :
            # try all the combinations of calves to get to the right distribution
            calf_combinations = list(combinations(self.calf_with_full_behaviours, nb_calf))
            for calf_combination in calf_combinations :
                total_proportion = sum([full_calf_proportions[calf] for calf in calf_combination])
                total_proportion = round(total_proportion, 0)
                # print(f"nb calf {nb_calf}, combination {calf_combination}, sum proportion : {total_proportion}")
                # chek if we get to the right distribution
                if total_proportion == self.val_prop*100 or total_proportion == (self.val_prop*100-1) or total_proportion == (self.val_prop*100+1) :
                    working_val_combinations.append(calf_combination)
                    if self.verbose :
                        print(f"---Val working combination {calf_combination}, sum proportion : {total_proportion}")
        
        if working_val_combinations == [] :
            raise ValueError("No combination of calves can be found to get to the right distribution for the validation set")
        
        # generating the possible combinations for the test and validation sets together
        unique_combinations = []
        
        for test_combination in working_test_combinations :
            for val_combination in working_val_combinations :
                # check if the combination are different (not the same calf in the test and validation sets)
                if len(set(test_combination).intersection(set(val_combination))) == 0 :
                    unique_combinations.append((test_combination, val_combination))
                
        print(f"\nNumber of possible combinations : {len(unique_combinations)}")
        # print(f"Example of possible combination : {unique_combinations[0]}")
        
        # select the combination with the least proportion difference to the train set
        diff_test_avgs_and_train = {}
        diff_val_avgs_and_train = {}
        total_diff_test_val_and_train = {}
        
        for test_combination, val_combination in unique_combinations :
            
            calf_to_not_use = list(test_combination) + list(val_combination)
            df_train_from_test_and_val = self.regrouped_dataset[~self.regrouped_dataset["calf_id"].isin(calf_to_not_use)]
            df_train_from_test_and_val_proportion = df_train_from_test_and_val['label'].value_counts(normalize = True)*100
            # print(f"\nTrain Test proportion :\n {df_train_from_test_and_val_proportion}")
            
            test_proportion = [self.regrouped_dataset[self.regrouped_dataset['calf_id'] == calf]['label'].value_counts(normalize = True)*100 for calf in test_combination]
            val_proportion = [self.regrouped_dataset[self.regrouped_dataset['calf_id'] == calf]['label'].value_counts(normalize = True)*100 for calf in val_combination]
            # print(f"Test proportion : {test_proportion}")
            
            test_proportion_behaviour = [ len(self.regrouped_dataset[self.regrouped_dataset['calf_id'] == calf]) for calf in test_combination]
            val_proportion_behaviour = [ len(self.regrouped_dataset[self.regrouped_dataset['calf_id'] == calf]) for calf in val_combination]
            # print(f"Test proportion : {test_proportion_behaviour}")
            
            # calculating the weighted average proportion for each behaviour
            test_weighted_avg_proportion = []
            for i in range(len(test_proportion)) :
                test_weighted_avg_proportion.append(test_proportion[i]*test_proportion_behaviour[i]/sum(test_proportion_behaviour))
            
            test_weighted_avg_proportion = sum(test_weighted_avg_proportion)
            # print(f"\nTest proportion weighted average for combination {test_combination} :\n {test_weighted_avg_proportion}")
            
            val_weighted_avg_proportion = []
            for i in range(len(val_proportion)) :
                val_weighted_avg_proportion.append(val_proportion[i]*val_proportion_behaviour[i]/sum(val_proportion_behaviour))
            
            val_weighted_avg_proportion = sum(val_weighted_avg_proportion)
            # print(f"\nVal proportion weighted average for combination {val_combination} :\n {val_weighted_avg_proportion}")

            # the difference between the weighted average proportion and the train proportion
            diff_test_avgs_and_train[test_combination]  = sum(abs(test_weighted_avg_proportion - df_train_from_test_and_val_proportion))/len(test_combination)
            # print(f"\n Sum of the diff between the train and TEST proportion weighted average for combination {test_combination} :\n {diff_test_avgs_and_train[test_combination]}")
            
            diff_val_avgs_and_train[val_combination]  = sum(abs(val_weighted_avg_proportion - df_train_from_test_and_val_proportion))/len(val_combination)
            # print(f"Sum of the diff between the train and VAL proportion weighted average for combination {val_combination} :\n {diff_val_avgs_and_train[val_combination]}")
            
            total_diff_test_val_and_train[(test_combination, val_combination)] = diff_test_avgs_and_train[test_combination] + diff_val_avgs_and_train[val_combination]
            
            if self.verbose :
                print(f"Sum of the diff between the train and TEST+VAL proportion weighted average for combination {(test_combination, val_combination)} : {total_diff_test_val_and_train[(test_combination, val_combination)]}")

        print(f"\nAll the differences between train and val+set sets : {total_diff_test_val_and_train}")
        
        # saving the results
        self.total_diff_test_val_and_train = total_diff_test_val_and_train
        
        # select the combination with the least difference
        best_combination = min(total_diff_test_val_and_train, key=total_diff_test_val_and_train.get)
        print(f"Best combination : {best_combination}")
        # print(type(best_combination))
        # print(best_combination[0])
        # print(best_combination[1][0])
    
        print("\n--- All the combinations have been tested ---")
        
        return best_combination
    
    
    def save_the_datasets (self) :
        """
        Save the datasets in the dataset folder depeending on the grouping of the original dataset
        """
        ##### to be ugraded
                
        test_calves, validation_calves = self.selected_test_val_calves
        calf_to_not_use = list(test_calves) + list(validation_calves)
        
        df_train = self.regrouped_dataset[~self.regrouped_dataset["calf_id"].isin(calf_to_not_use)]
        df_test = self.regrouped_dataset[self.regrouped_dataset["calf_id"].isin(test_calves)]
        df_val = self.regrouped_dataset[self.regrouped_dataset["calf_id"].isin(validation_calves)]
        
        # update the proportions of the different sets
        self.train_prop = round(len(df_train)/len(self.regrouped_dataset)*100, 1)
        self.test_prop = round(len(df_test)/len(self.regrouped_dataset)*100, 1)
        self.val_prop = round(len(df_val)/len(self.regrouped_dataset)*100, 1)
        
        if len(self.grouping_dict.keys()) >= 6 :
            
            df_train.to_csv(f"{self.path_output}/train/{date.today()}_DETAILED_train_{self.train_prop}_dataset.csv", index=False)
            df_test.to_csv(f"{self.path_output}/test/{date.today()}_DETAILED_test_{self.test_prop}_dataset.csv", index=False)
            df_val.to_csv(f"{self.path_output}/validation/{date.today()}_DETAILED_val_{self.val_prop}_dataset.csv", index=False)
            
        if len(self.grouping_dict.keys()) < 6 :
            
            df_train.to_csv(f"{self.path_output}/train/{date.today()}_STATE_train_{self.train_prop}_dataset.csv", index=False)
            df_test.to_csv(f"{self.path_output}/test/{date.today()}_STATE_test_{self.test_prop}_dataset.csv", index=False)
            df_val.to_csv(f"{self.path_output}/validation/{date.today()}_STATE_val_{self.val_prop}_dataset.csv", index=False)
        print("\n--- The datasets have been saved ---")
    

def make_parser() :
    """
    Parser function for the arguments of the pipeline
    Inputs :
    Returns : the parser itself
    """
    parser = argparse.ArgumentParser(description='DTW_KNN algorithm')
    parser.add_argument('--size_of_train', type=float, default=0.7, help='split (in pourcentages) for the train set (default: 0.7)')
    parser.add_argument('--size_of_test', type=float, default=0.15, help='split (in pourcentages) for the train set (default: 0.15)')
    parser.add_argument('--size_of_val', type=float, default=0.15, help='split (in pourcentages) for the train set (default: 0.15)')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose (default: False)')
    
    return parser

##### Definition of the groups
beha_groups_short = { 'moving': ["run|", "run|play", "run|not play", 
                                'walking|', 'walking|Forward', 'walking|Backward', 
                                'play|jump|standing', "sniff|walking"], 
                        'idling' : ["sniff|standing", "standing|", "sniff|"],
                       'lying': ["lying|"],
                       'drinking_milk': ["drinking|", 'drinking|milk'], 
                       'eating' : ['eating|bedding|standing', 'eating|concentrates|standing','eating|forage|standing']}
                       

beha_groups_detailed = {'running': ["run|", "run|play", "run|not play",'play|jump|standing'],
                        "walking" :['walking|', 'walking|Forward', 'walking|Backward',  "sniff|walking"],
                        'idling' : ["sniff|standing", "standing|", "sniff|"],
                       'lying': ["lying|"],
                       'drinking_milk': ["drinking|", 'drinking|milk'], 
                       'eating' : ['eating|concentrates|standing'],
                       'grooming' : ['grooming|', 'grooming|lying','grooming|standing'],
                       'transition' : ['lying-down|', 'rising|'],
                       'oral manipulation of the pen' : ["oral manipulation of pen|", 'oral manipulation of pen|None',
                                        'oral manipulation of pen|standing', 'oral manipulation of the pen|standing'],
                       "social interaction" : ['social interaction|None|None', 'social interaction|None|standing', 
                                        'social interaction|groom|lying','social interaction|groom|standing','social interaction|nudge|standing',
                                        'social interaction|sniff|lying', 'social interaction|sniff|standing']}
    
path_outputs = "/home/Calf Behaviour Classification/PROD/Files/datasets"


if __name__ == '__main__':
    start = time.time()
    
    # get the arguments from the cd
    parser = make_parser()
    args = parser.parse_args()
    
    size_of_train = args.size_of_train
    size_of_test = args.size_of_test
    size_of_val = args.size_of_val
    verbose_ = args.verbose
    
    path_dataset = "/home/Calf Behaviour Classification/PROD/Files/dataset/whole data/2023-08-10_LONG_dataset.csv"
    #### Loading the datasets
    df_dataset = pd.read_csv(path_dataset, engine='pyarrow')
    
    #splitting the dataset
    dataset_split_short = Dataset_split(df_dataset, beha_groups_short, path_outputs, \
                                        train_prop = size_of_train, test_prop = size_of_test, validation_prop = size_of_val, \
                                        verbose = verbose_)
    
    dataset_split_detailed = Dataset_split(df_dataset, beha_groups_detailed, path_outputs, \
                                            train_prop = size_of_train, test_prop = size_of_test, validation_prop = size_of_val, \
                                            verbose = verbose_)