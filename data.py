import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import chardet

import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

import seaborn as sns


#%% Data Loading and Saving

def create_folder(input_folder_path):
    assert os.path.exists(input_folder_path) is False, "Folder already exists"

    os.makedirs(input_folder_path)

    return True

def load_file(input_file_path):
    accepted_file_types = ['.txt', '.json', '.pickle', '.csv']
    _, file_extension = os.path.splitext(input_file_path)
    assert file_extension in accepted_file_types, "Cannot load this file type"
    
    if file_extension == '.txt':
        with open(input_file_path, encoding='utf8') as file:
            content = file.read()
        return content
    
    elif file_extension == '.json':
        with open(input_file_path) as file:
            content = json.load(file)
        return content

    elif file_extension == '.pickle':        
        with open(input_file_path, 'rb') as file:
            content = pickle.load(file)
        return content
    
    elif file_extension == '.csv':
        try:
            content = pd.read_csv(input_file_path)
            return content
            
        except Exception as e:
            print(e)
            with open(file, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(10000))
            
                # Note: low memory option will enable columns with mixed data types to be asserted later
                content = pd.read_csv(file, encoding=result['encoding'], low_memory=False)
                return content

    else:
        print("Please select one of: {} file types".format(accepted_file_types))


def save_file(output_object, output_file_path, overwrite=False):
    accepted_file_types = ['.txt', '.json', '.pickle', '.csv']
    _, file_extension = os.path.splitext(output_file_path)
    assert file_extension in accepted_file_types, "Please select correct file type"
    
    if (overwrite is False) & (os.path.exists(output_file_path)):
        raise Exception("File '{}' exists. Please either set overwrite=True or rename the file.".format(
            output_file_path))
    
    else:
        if file_extension == '.txt':
            with open(output_file_path, 'w', encoding='utf8') as output_file:
                output_file.write(output_object)
            
        elif file_extension == '.json':  
            with open(output_file_path, 'w') as output_file:
                json.dump(output_object, output_file)
        
        elif file_extension == '.pickle':  
            with open(output_file_path, "wb") as output_file:
                pickle.dump(output_object, output_file)
        
        elif file_extension == '.csv':
            output_object.to_csv(output_file_path, ignore_index=True)
            
        else:
            print("Please select one of: {} file types".format(accepted_file_types))
        
        print("File saved successfuly at {}".format(output_file_path))

#%% Train Validation Test Split / Cross Validation

class ModelData():
    def __init__(self, input_df, target_name, split_ratios):
        """
        input_df: full dataframe
        target_name: column name of target variable for prediction
        split ratios: expects a dictionary of the splits as follows:
            {'train' : 0.60,
             'validation' : 0.20,
             'test' : 0.20}
        """
        len_input_df = len(input_df)
        random_idx = np.random.choice(
            list(range(len_input_df)), 
            len_input_df,
            replace=False)
        
        train_idx = int(split_ratios['train'] * len_input_df)
        
        if ('validation' in split_ratios) and (split_ratios['validation'] > 0):
            val_idx = int(split_ratios['train'] * len_input_df) + int(split_ratios['validation'] * len_input_df)
            
            train_set = random_idx[:train_idx]
            val_set = random_idx[train_idx:val_idx]
            test_set = random_idx[val_idx:]
                
        else:
            train_set = random_idx[:train_idx]
            test_set = random_idx[train_idx:]
        
        self.input_df = input_df
        self.target_name = target_name
        self.split_ratios = split_ratios
        
        self.X_train = input_df.loc[train_set, input_df.columns != target_name].reset_index(drop=True)
        self.y_train = input_df.loc[train_set, target_name].reset_index(drop=True)
        self.X_test = input_df.loc[test_set, input_df.columns != target_name].reset_index(drop=True)
        self.y_test = input_df.loc[test_set, target_name].reset_index(drop=True)
        
        if ('validation' in split_ratios) and (split_ratios['validation'] > 0):
            self.X_val = input_df.loc[val_set, input_df.columns != target_name].reset_index(drop=True)
            self.y_val = input_df.loc[val_set, target_name].reset_index(drop=True)
        else:
            self.X_val = None
            self.y_val = None
            
#%% Handle Duplicate Values

def get_duplicate_rows(input_df, column_subset=['Id']):
    """Returns a dataframe containing the duplicated rows"""
    return input_df.loc[input_df.duplicated(subset=column_subset, keep='first')]

# Get duplicate rows then remove them from dataframe
# duplicate_rows = get_duplicate_rows(input_df, column_subset)
# df = df.drop(duplicate_rows.index).reset_index(drop=True)

#%% Handle Null Values

def prop_column_null(input_df):
    """Returns the proportion of null values by column"""
    return (input_df.isnull().sum(axis = 0) / len(input_df)).sort_values(ascending=False)

def prop_row_null(input_df):
    """Returns the proportion of null values by row"""
    return (input_df.isnull().sum(axis = 1) / len(input_df.columns)).sort_values(ascending=False)

class SimpleImputer():
    def __init__(self):
        self.model_parameters = {}
    
    def fit(self, input_series, strategy='mean'):
        valid_strategies = ['mean', 'median', 'mode', 'unknown']
        assert strategy in valid_strategies

        if strategy == 'mean':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : input_series.mean()}
            
        elif strategy == 'median':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : input_series.median()}
            
        elif strategy == 'mode':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : input_series.mode().values[0]}
        
        elif strategy == 'unknown':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : 'unknown'}
        
        else:
            print("Please select one of: {} as imputation strategy".format(valid_strategies))
    
        return self.model_parameters[input_series.name]
    
    def transform(self, input_series):
        return input_series.fillna(self.model_parameters[input_series.name]['value'])

    def apply_constant_value(self, input_series, constant_value):
        self.model_parameters[input_series.name] = {'strategy' : 'constant value', 'value' : constant_value}
        return input_series.fillna(constant_value)
        

#%% Handle imbalanced datasets

def balance_data(input_X_train, input_y_train, target_ratio, approach='SMOTE'):
    assert approach in ['random_undersampling', 'random_oversampling', 'SMOTE']
    
    if approach == 'random_undersampling':
        n_positive = np.sum(input_y_train)
        n_total =  int(n_positive / target_ratio)
        n_negative = n_total - n_positive
        model = RandomUnderSampler(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
    
    elif approach == 'random_oversampling':
        n_negative = np.sum(DF.y_train == 0)
        n_positive = int((n_negative / (1 - target_ratio)) - n_negative)
        model = RandomOverSampler(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
        
    elif approach == 'SMOTE':
        # SMOTE -- Need to remove null values and convert categorial variables before using
        assert len(input_X_train.select_dtypes('object').columns) == 0, "Need to remove categorical features before using SMOTE"
        assert np.sum(np.sum(input_X_train.isnull())) == 0, "Need to remove null values before using SMOTE"

        n_negative = np.sum(input_y_train == 0)
        n_positive = int((n_negative / (1 - target_ratio)) - n_negative)
        model = SMOTE(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
    
    X_resampled, y_resampled = model.fit_resample(input_X_train, input_y_train)

    print("Resampling complete.")
    print("{} records before | {} records after".format(len(input_X_train), len(X_resampled)))
    print("{:.1%} positive | {:.1%} positive".format(
        input_y_train.sum() / len(input_y_train), y_resampled.sum() / len(y_resampled)))

    return X_resampled, y_resampled

#%% Feature Engineering

class StandardScaler():
    def __init__(self):
        self.model_parameters = {}
        
    def fit(self, input_series):
        self.model_parameters[input_series.name] = {'mean': input_series.mean(), 'std_dev':input_series.std()}
        return self.model_parameters[input_series.name]
        
    def transform(self, input_series):
        return (input_series - self.model_parameters[input_series.name]['mean']) / self.model_parameters[input_series.name]['std_dev']


class MeanEncoder():
    def __init__(self):
        self.model_parameters = {}
    
    def fit(self, input_series, target_output):
        merged_df = pd.merge(input_series, target_output, how='left', left_index=True, right_index=True)
        mean_target_value = merged_df.groupby(input_series.name).mean()
        self.model_parameters[input_series.name] = mean_target_value[target_output.name].to_dict()
        return self.model_parameters[input_series.name]
        
    def transform(self, input_series):
        return input_series.map(self.model_parameters[input_series.name])

class BinEncoder():
    def __init__(self):
        self.model_parameters = {}
        
    def fit(self, input_series, n_bins=5):
        self.model_parameters[input_series.name] = (input_series.size/float(n_bins)) * np.arange(1, n_bins+1)
        return self.model_parameters[input_series.name]
    
    def transform(self, input_series):
        idx = self.model_parameters[input_series.name].searchsorted(np.arange(input_series.size))
        return idx[input_series.argsort().argsort()]

def one_hot_encode_column(input_series):
    one_hot_df = pd.get_dummies(input_series)
    one_hot_df.columns = [input_series.name+': '+x for x in one_hot_df.columns]
    
    return one_hot_df
    
def log_transform_column(input_series):
    assert input_series.dtype != 'O'
    
    return np.log(input_series)

# Other feature engineering
def convert_utc_to_dt(input_utc):
    """Converts UTC to datetime and then provides a formatted date time"""
    dt = datetime.fromtimestamp(input_utc)

    return {'converted_date' : dt, 'converted_date_string' : dt.strftime("%d-%b-%Y - %I:%M %p")}


#%% TODO: 
# - Can add enhanced imputation strategies (KNN, Decision Tree, etc.)




# Does not require state for test transform




#%% ------------------------- Archive ----------------------------

# def bin_feats(input_df, features_to_bin, n_bins=10):
#     for feature in features_to_bin:
#         input_df[feature] = equal_bin(input_df[feature], n_bins)


# def encode_cat_vars(X_train_input, X_test_input, cat_encoding_dict):
#     for feature, encoding in cat_encoding_dict.items():
#         if encoding == 'one-hot':
#             X_train_output, X_test_output = one_hot_encode(X_train_input[feature], X_test_input[feature])
#             X_train_input = X_train_input.join(X_train_output)
#             X_test_input = X_test_input.join(X_test_output)

#             X_train_input.drop(feature,axis=1,inplace=True)
#             X_test_input.drop(feature,axis=1,inplace=True)

#         elif encoding == 'mean-encode':
#             X_train_input[feature], X_test_input[feature] = mean_encode(X_train_input[feature], X_test_input[feature])
            
#         # Ensure all columns are shared between train / test, if not then add zero column to test set
#         for col in X_train_input.columns:
#             if col not in X_test_input.columns:
#                 X_test_input[col] = [0] * len(X_test_input)  
#     return X_train_input, X_test_input