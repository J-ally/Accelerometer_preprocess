# -*- coding: utf-8 -*-
"""
Created on August 15 2023
@author: Joseph
"""

from cmath import nan
import pandas as pd
import numpy as np
import math
from scipy import signal
from scipy.fft import fft
from scipy import stats
import statistics as sts


###############################################################################
#                               IMPORT TESTING                                #
###############################################################################
#  The following function can be used to test if the file is imported correctly:

def test_import () -> bool:
    """
    Returns True if the file i
    s imported correctly.
    Returns:
        True (bool): True if the file is imported correctly.
    """
    print("Features_calc.py imported successfully")
    return True


###############################################################################
#                            FEATURES CALCULATIONS                            #
###############################################################################

class Features_Calculations () :
    """
    Features calculations class
    All axis can be permmuted to fit the orientation of the censor
    Inputs :
        - dataset : pandas dataframe containing the data
            Need to have three columns : accx, accy, accz, and the following features will be calculated :
                - amag : magnitude of the acceleration
                - pitch : pitch angle
                - roll : roll angle
                - OBDA : OBDA
        
    """
    
    def __init__ (self, dataset : pd.DataFrame, features_list : list, verbose = True) :
        self.dataset = dataset
        self.original_dataset = dataset.copy()
        self.features_list = features_list
        self.verbose = verbose
        self.added_features = []
        

        # self.calculate_amag() # ama is calculated in our case
        self.calculate_obda()
        self.calculate_pitch_roll()
        
                
        self.dict_features = {feature : [] for feature in self.features_list }
        
        # adding the metadata to the dict_features
        
        ts_metadata = {"calf_id" : self.dataset.iloc[0]["calf_id"],
                        "label" : self.dataset.iloc[0]["label"],
                        "start_time" : self.dataset.iloc[0]["adjusted_DateTime"],
                        "end_time" : self.dataset.iloc[-1]["adjusted_DateTime"] }
        
        self.dict_features = {**ts_metadata, **self.dict_features}
        # print(type(ts_metadata))
        # print(type(self.dict_features))
        # calculating the features
        for feature in self.features_list :
            feature_function = feature.split("_")[0]
            feature_name_for_calculations = feature.split("_")[1]
            
            if feature_function == "mean" :
                self.dict_features[feature] = self.calculate_mean(feature_name_for_calculations)
            elif feature_function == "std" :
                self.dict_features[feature] = self.calculate_std(feature_name_for_calculations)
            elif feature_function == "RMS" :
                self.dict_features[feature] = self.calculate_RMS(feature_name_for_calculations)
            elif feature_function == "max" :
                self.dict_features[feature] = self.calculate_max(feature_name_for_calculations)
            elif feature_function == "min" :
                self.dict_features[feature] = self.calculate_min(feature_name_for_calculations)
            elif feature_function == "range" :
                self.dict_features[feature] = self.calculate_range(feature_name_for_calculations)
            elif feature_function == "median" :
                self.dict_features[feature] = self.calculate_median(feature_name_for_calculations)
            elif feature_function == "q1" :
                self.dict_features[feature] = self.calculate_q1(feature_name_for_calculations)
            elif feature_function == "q3" :
                self.dict_features[feature] = self.calculate_q3(feature_name_for_calculations)
            elif feature_function == "interQuartile" :
                self.dict_features[feature] = self.calculate_interQuartile(feature_name_for_calculations)
            elif feature_function == "skewness" :
                self.dict_features[feature] = self.calculate_skewness(feature_name_for_calculations)
            elif feature_function == "kurtosis" :
                self.dict_features[feature] = self.calculate_kurtosis(feature_name_for_calculations)
            elif feature_function == "correlation" :
                self.dict_features[feature] = self.calculate_correlation("acc"+feature_name_for_calculations[0], "acc"+feature_name_for_calculations[1])
            elif feature_function == "MV" :
                self.dict_features[feature] = self.calculate_motion_variation(feature_name_for_calculations)
            elif feature_function == "spectral" :
                self.dict_features[feature] = self.calculate_entropy()
                

            
    def generate_error_message (self, feature_name : str, features_researched : str) :
        print(f"Calculating {feature_name} KeyError, following feature not found in the dataset : {features_researched}" )
        print(f"The columns of the dataset are : {self.dataset.columns}")
             
             
    def calculate_amag (self) :
        if "amag" in self.dataset.columns :
            print("amag already calculated")
            pass
        
        try :
            self.dataset["amag"] = np.sqrt(self.dataset["accx"]**2 + self.dataset["accy"]**2 + self.dataset["accz"]**2)
            self.added_features.append("amag")
        except KeyError:
            self.generate_error_message("amag", "accx, accy, accz")
    
    
    def calculate_pitch_roll (self) :
        """
        Calculates the pitch and roll angles from the accelerometer data
        Permutations are applied to the axis before the calculation
        in our case, the permutations are : 
                        X -> Y
                        Y -> Z
                        Z -> X
                   Normal -> Our axis
        """
        
        if "pitch" in self.dataset.columns and "roll" in self.dataset.columns :
            print("pitch and roll already calculated")
            pass
        
        try :
            pitch = []
            roll = []
            for row in self.dataset.itertuples() :
                # supposing that X is your side-to-side axis
                # originally : pitch.append(180*math.atan2(row.accx,np.sqrt(row.accy**2 + row.accz**2))/math.pi) 
                pitch.append(180*math.atan2(row.accy,np.sqrt(row.accz**2 + row.accx**2))/math.pi)
                # supposing that Y is your front-to-back axis
                # originally : roll.append(180*math.atan2(row.accy,np.sqrt(row.accx**2 + row.accz**2))/math.pi)
                roll.append(180*math.atan2(row.accz,np.sqrt(row.accy**2 + row.accx**2))/math.pi)
                
            self.dataset["pitch"] = pitch
            self.dataset["roll"] = roll
            self.added_features.append("pitch")
            self.added_features.append("roll")
        except KeyError:
            self.generate_error_message("pitch and roll", "accx, accy, accz")


    def calculate_obda (self) :
        if "OBDA" in self.dataset.columns :
            print("OBDA already calculated")
            pass
        
        try :
            df_dynamic = self.dataset[["accx", "accy", "accz"]].copy()
            sos_dynamic = signal.butter(6, 0.3, 'high', output = 'sos', analog=False)
            
            df_dynamic['dynamic_x'] = signal.sosfilt(sos_dynamic, df_dynamic['accx'])
            df_dynamic['dynamic_y'] = signal.sosfilt(sos_dynamic, df_dynamic['accy'])
            df_dynamic['dynamic_z'] = signal.sosfilt(sos_dynamic, df_dynamic['accz'])
        
            self.dataset['OBDA'] = abs(df_dynamic['dynamic_x'])+abs(df_dynamic['dynamic_y'])+abs(df_dynamic['dynamic_z'])
            self.added_features.append("OBDA")
            
        except KeyError :
            self.generate_error_message("OBDA", "accx, accy, accz")
        
            

    def calculate_mean (self, feature_colname : str) :
        if f"mean_{feature_colname}" in self.dataset.columns :
            print(f"mean_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"mean_{feature_colname}"] = self.dataset[feature_colname].mean()
            self.added_features.append(f"mean_{feature_colname}")
            return self.dataset[f"mean_{feature_colname}"].values[0]
            
        except KeyError:
            self.generate_error_message(f"mean of {feature_colname}", feature_colname) 



    def calculate_std (self, feature_colname : str) :
        if f"std_{feature_colname}" in self.dataset.columns :
            print(f"std_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"std_{feature_colname}"] = self.dataset[feature_colname].std()
            self.added_features.append(f"std_{feature_colname}")
            return self.dataset[f"std_{feature_colname}"].values[0]
            
        except KeyError:
            self.generate_error_message(f"std of {feature_colname}", feature_colname) 

    
    
    def calculate_RMS (self, feature_colname : str) :
        if f"RMS_{feature_colname}" in self.dataset.columns :
            print(f"RMS_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"RMS_{feature_colname}"] = np.sqrt(self.dataset[f"mean_{feature_colname}"]**2)
            self.added_features.append(f"RMS_{feature_colname}")
            return self.dataset[f"RMS_{feature_colname}"].values[0]
            
        except KeyError:
            self.generate_error_message(f"RMS of {feature_colname}", feature_colname) 

    

    def calculate_min (self, feature_colname : str) :
        if f"min_{feature_colname}" in self.dataset.columns :
            print(f"min_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"min_{feature_colname}"] = self.dataset[feature_colname].min()
            self.added_features.append(f"min_{feature_colname}")
            return self.dataset[f"min_{feature_colname}"].values[0]
        except KeyError:
            self.generate_error_message(f"min of {feature_colname}", feature_colname)
    
    
    def calculate_max (self, feature_colname : str) :
        if f"max_{feature_colname}" in self.dataset.columns :
            print(f"max_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"max_{feature_colname}"] = self.dataset[feature_colname].max()
            self.added_features.append(f"max_{feature_colname}")
            return self.dataset[f"max_{feature_colname}"].values[0]
            
        except KeyError:
            self.generate_error_message(f"max of {feature_colname}", feature_colname)
            
    
    def calculate_range (self, feature_colname : str) :
        if f"range_{feature_colname}" in self.dataset.columns :
            print(f"range_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"range_{feature_colname}"] = self.dataset[f"max_{feature_colname}"] - self.dataset[f"min_{feature_colname}"]
            self.added_features.append(f"range_{feature_colname}")
            return self.dataset[f"range_{feature_colname}"].values[0]
            
        except KeyError:
            self.generate_error_message(f"range of {feature_colname}", feature_colname)
            
    

    def calculate_median (self, feature_colname : str) :
        if f"median_{feature_colname}" in self.dataset.columns :
            print(f"median_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"median_{feature_colname}"] = self.dataset[feature_colname].median()
            self.added_features.append(f"median_{feature_colname}")
            return self.dataset[f"median_{feature_colname}"].values[0]
        
        except KeyError:
            self.generate_error_message(f"median of {feature_colname}", feature_colname)
            
            
    def calculate_q3 (self, feature_colname : str) :
        if f"q3_{feature_colname}" in self.dataset.columns :
            print(f"q3_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"q3_{feature_colname}"] = self.dataset[feature_colname].quantile(0.75)
            self.added_features.append(f"q3_{feature_colname}")
            return self.dataset[f"q3_{feature_colname}"].values[0]
            
        except KeyError:
            self.generate_error_message(f"q3 of {feature_colname}", feature_colname)
            
    
    def calculate_q1 (self, feature_colname : str) :
        if f"q1_{feature_colname}" in self.dataset.columns :
            print(f"q1_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"q1_{feature_colname}"] = self.dataset[feature_colname].quantile(0.25)
            self.added_features.append(f"q1_{feature_colname}")
            return self.dataset[f"q1_{feature_colname}"].values[0]
            
        except KeyError:
            self.generate_error_message(f"q1 of {feature_colname}", feature_colname)
            
            
    def calculate_interQuartile (self, feature_colname : str) :
        if f"interQuartile_{feature_colname}" in self.dataset.columns :
            print(f"interQuartile_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"interQuartile_{feature_colname}"] = self.dataset[f"q3_{feature_colname}"] - self.dataset[f"q1_{feature_colname}"]
            self.added_features.append(f"interQuartile_{feature_colname}")
            return self.dataset[f"interQuartile_{feature_colname}"].values[0]
            
        except KeyError:
            self.generate_error_message(f"interQuartile of {feature_colname}", feature_colname)
                
    
    def calculate_skewness (self, feature_colname : str) :
        if f"skewness_{feature_colname}" in self.dataset.columns :
            print(f"skewness_{feature_colname}, kurtosis_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"skewness_{feature_colname}"] = self.dataset[feature_colname].skew()
            self.added_features.append(f"skewness_{feature_colname}")
            return self.dataset[f"skewness_{feature_colname}"].values[0]
        except KeyError:
            self.generate_error_message(f"skewness of {feature_colname}", feature_colname)

    
    def calculate_kurtosis (self, feature_colname : str) :
        if f"kurtosis_{feature_colname}" in self.dataset.columns :
            print(f"kurtosis_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"kurtosis_{feature_colname}"] = self.dataset[feature_colname].kurtosis()
            self.added_features.append(f"kurtosis_{feature_colname}")
            return self.dataset[f"kurtosis_{feature_colname}"].values[0]
        except KeyError:
            self.generate_error_message(f"kurtosis of {feature_colname}", feature_colname)
        
            
    def calculate_entropy(self) :
        if "spectral_entropy" in self.dataset.columns :
            print(f"spectral_entropy already calculated")
            pass
         
        # entropy is calculated on the magnitude of the acceleration
        try :
            window_Amag = self.dataset["amag"].values
        except KeyError:
            print(f"Error : amag not found in the dataset")
            print(f"Trying for Amag")
            try :
                window_Amag = self.dataset["Amag"].values
            except KeyError:
                self.generate_error_message("amag or Amag", "amag or Amag")
                
        # fft using Hamming Window
        AmagFFT = fft(window_Amag*np.hamming(len(window_Amag)))

        # Single-sided spectrum = module fft
        absAmagFFT = abs(AmagFFT[1:math.floor(len(window_Amag)/2+1)])**2

        WeightAbsAmagFFT = absAmagFFT/sum(absAmagFFT)
        SpectralEntropy = -sum(WeightAbsAmagFFT * np.log(WeightAbsAmagFFT))
        
        self.dataset["spectral_entropy"] = SpectralEntropy
        self.added_features.append("spectral_entropy")
        return SpectralEntropy
        
        
    def calculate_motion_variation(self, feature_colname : str) :
        if f"MV_{feature_colname}" in self.dataset.columns :
            print(f"MV_{feature_colname} already calculated")
            pass
        
        try :
            self.dataset[f"MV_{feature_colname}"] = abs(self.dataset[feature_colname].diff()).mean()
            self.added_features.append(f"MV_{feature_colname}")
            return self.dataset[f"MV_{feature_colname}"].values[0]
        except KeyError:
            self.generate_error_message(f"motion variation of {feature_colname}", feature_colname)
    
    
    def calculate_correlation (self, feature_1 : str, feature_2 : str) :
        if f"correlation_{feature_1[-1]}{feature_2[-1]}" in self.dataset.columns :
            print(f"correlation_{feature_1[-1]}{feature_2[-1]} already calculated")
            pass
        
        try :
            self.dataset[f"correlation_{feature_1[-1]}{feature_2[-1]}"] = self.dataset[feature_1].corr(self.dataset[feature_2])
            self.added_features.append(f"correlation_{feature_1[-1]}{feature_2[-1]}")
            return self.dataset[f"correlation_{feature_1[-1]}{feature_2[-1]}"].values[0]
        except KeyError:
            self.generate_error_message(f"correlation_{feature_1[-1]}{feature_2[-1]}", "feature_1 or feature_2")
    
    
    def return_dataset (self) -> pd.DataFrame :
        return self.dataset
    
    def get_feature_dict (self) -> dict :
        return self.dict_features
    

###############################################################################
#                                  TESTING                                    #
###############################################################################

# X_axis = [np.random.randint(0,10) for i in range(100)]
# Y_axis = [np.random.randint(0,10) for i in range(100)]
# Z_axis = [np.random.randint(0,10) for i in range(100)]


# df_random = pd.DataFrame({"mean_accx" : X_axis, "accx" : X_axis, "accy" : Y_axis, "accz" : Z_axis})

# A = Features_Calculations(df_random, {"accx" : "accy", "accy" : "accz", "accz" : "accx"})

# A.calculate_mean_std_RMS("accx")
# A.calculate_min_max_range("accx")
# A.calculate_median("accx")
# A.calculate_quartiles("accx")
# A.calculate_correlation("accx", "accy")

# print(A.added_features)

# print(A.return_dataset().head())
