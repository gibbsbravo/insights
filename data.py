import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import chardet

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
                json.dump(output_object, output_file, indent=4)
        
        elif file_extension == '.pickle':  
            with open(output_file_path, "wb") as output_file:
                pickle.dump(output_object, output_file)
        
        elif file_extension == '.csv':
            output_object.to_csv(output_file_path, ignore_index=True)
            
        else:
            print("Please select one of: {} file types".format(accepted_file_types))
        
        print("File saved successfuly at {}".format(output_file_path))

def sample_data(input_df, fraction=0.40):
    return pd.sample(frac=fraction, replace=False, random_state=34)

def create_parameters_template(input_X_train, regression_objective=True, 
                               output_file_path='data/pipeline_parameters.json', overwrite=False):
    pipeline_parameters = {
        'removed_features' : [],
        'imputing_strategies' : {col : {'strategy' : 'mode', 'constant_value' : None, 'model' : None} 
                                      for col in input_X_train.loc[:, input_X_train.isna().any()]},
        'categorical_encodings' : {col : {'encoding' : 'mean', 'model' : None} 
                                      for col in input_X_train.select_dtypes('object').columns},
        'include_engineered_feats' : True,
        'include_clustered_feats' : True,
        'cluster_models' : [
            {'cluster_name' : 'all_features_kmeans',
             'model_name' : 'Kmeans',
             'features' : [],
             'n_clusters' : 5,
             'model' : None},
            {'cluster_name' : 'neighborhood_kmeans',
             'model_name' : 'Kmeans',
             'features' : [],
             'n_clusters' : 5,
             'model' : None}]
        }
    
    if regression_objective:
        pipeline_parameters['models'] = {
            'LGBM' : {'default_hyperparameters' : {'num_leaves' : 10},
                      'gridsearch_hyperparameters' : {'num_leaves':[5, 15, 30, 60, 90]}},
            
            'Linear' : {'default_hyperparameters' : None,
                      'gridsearch_hyperparameters' : None},
            
            'Ridge' : {'default_hyperparameters' : {'alpha' : 1},
                      'gridsearch_hyperparameters' : {'alpha':[0.01, 0.1, 1, 10, 100]}},
            
            
            'Random Forest' : {'default_hyperparameters' : {'max_depth' : 10},
                      'gridsearch_hyperparameters' : {'max_depth':[1, 2, 8, 10, 20]}},
        
            
            'KNN' : {'default_hyperparameters' : {'n_neighbors' : 5},
                      'gridsearch_hyperparameters' : None},
            }
        
    else:
        pipeline_parameters['models'] = {
            'LGBM' : {'default_hyperparameters' : {'num_leaves' : 10},
                      'gridsearch_hyperparameters' : {'num_leaves':[5, 15, 30, 60, 90]}},
            
            'Logistic Regression' : {'default_hyperparameters' : {'C' : 1, 'penalty' : 'l2'},
                      'gridsearch_hyperparameters' : {'C':[0.01, 0.1, 1, 10, 100]}},
            
            'Random Forest' : {'default_hyperparameters' : {'max_depth' : 10},
                      'gridsearch_hyperparameters' : {'max_depth':[1, 2, 8, 10, 20]}},
        
            'SVM' : {'default_hyperparameters' : {'C' : 1},
                      'gridsearch_hyperparameters' : {'C':[0.01, 0.1, 10],
                                                      'kernel':['linear', 'rbf']}},
            
            'KNN' : {'default_hyperparameters' : {'n_neighbors' : 5},
                      'gridsearch_hyperparameters' : None},
            }
        
    save_file(pipeline_parameters, output_file_path, overwrite)
    
    return pipeline_parameters

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
            
            self.X_full_train = pd.concat([self.X_train, self.X_val], axis=0)
            self.y_full_train = pd.concat([self.y_train, self.y_val], axis=0)
        else:
            self.X_val = None
            self.y_val = None
            
            self.X_full_train = self.X_train.copy()
            self.y_full_train = self.y_train.copy()
            
#%% Handle Duplicate Values

def get_duplicate_rows(input_df, column_subset=None):
    """Returns a dataframe containing the duplicated rows"""
    if column_subset is None:
        return input_df.loc[input_df.duplicated(keep='first')]
    else:
        return input_df.loc[input_df.duplicated(subset=column_subset, keep='first')]

def drop_duplicate_rows(input_df, column_subset=None):
    output_df = input_df.copy()
    # Get duplicate rows then remove them from dataframe
    duplicate_rows = get_duplicate_rows(output_df, column_subset)
    output_df = output_df.drop(duplicate_rows.index).reset_index(drop=True)
    
    return output_df 

def drop_columns(input_df, columns):
    output_df = input_df.copy()
    output_df.drop(columns, axis=1, inplace=True)
    return output_df

#%% Handle Null Values

class SimpleImputer():
    def __init__(self):
        self.model_parameters = {}
    
    def fit(self, input_series, strategy='mean', constant_value=None):
        valid_strategies = ['mean', 'median', 'mode', 'unknown', 'constant_value']
        assert strategy in valid_strategies

        if strategy == 'mean':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : input_series.mean()}
            
        elif strategy == 'median':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : input_series.median()}
            
        elif strategy == 'mode':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : input_series.mode().values[0]}
        
        elif strategy == 'unknown':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : 'unknown'}
            
        elif strategy == 'constant_value':
            self.model_parameters[input_series.name] = {'strategy' : strategy, 'value' : constant_value}
        
        else:
            print("Please select one of: {} as imputation strategy".format(valid_strategies))
    
        return self.model_parameters[input_series.name]
    
    def transform(self, input_series):
        return input_series.fillna(self.model_parameters[input_series.name]['value'])
    
    def fit_df(self, input_df, strategy='mean'):
        for col in input_df:
            self.fit(input_df[col], strategy)
        
        return self.model_parameters
    
    def transform_df(self, input_df):
        output_df = input_df.copy()
        for col in output_df:
            output_df[col] = self.transform(output_df[col])
        
        return output_df
    
    def fit_transform_df(self, input_df, strategy='mean'):
        output_df = input_df.copy()
        for col in output_df:
            self.fit(output_df[col], strategy)
            output_df[col] = self.transform(output_df[col])
        
        return output_df

def fit_null_value_imputing(input_X_train, imputing_strategies_dict, verbose=True):
    removed_features = []
    for col_name in imputing_strategies_dict:
        if col_name not in input_X_train.columns:
            removed_features.append(col_name)
        else:
            simple_imputer = SimpleImputer()
            simple_imputer.fit(
                input_X_train[col_name],
                strategy=imputing_strategies_dict[col_name]['strategy'],
                constant_value=imputing_strategies_dict[col_name]['constant_value'])
            imputing_strategies_dict[col_name]['model'] = simple_imputer
    
    if (len(removed_features) > 0) & (verbose):
        print("Null value fitting warning: can't fit {} columns as they have been removed".format(removed_features))

    return imputing_strategies_dict

def transform_null_value_imputing(input_df, imputing_strategies_dict, verbose=True):
    output_df = input_df.copy()
    removed_features = []
    
    for col_name in imputing_strategies_dict:
        if col_name not in input_df.columns:
            removed_features.append(col_name)
        else:
            simple_imputer = imputing_strategies_dict[col_name]['model']
            output_df[col_name] = simple_imputer.transform(output_df[col_name])
    
    if (len(removed_features) > 0) & (verbose):
        print("Null value transform warning: can't transform {} columns as they have been removed".format(removed_features))
    
    return output_df

def align_feature_cols(input_X_test, input_train_cols):
    output_X_test = input_X_test.copy()
    cols_not_in_pred = set(input_train_cols) - set(output_X_test.columns)
    
    for col in cols_not_in_pred:
        output_X_test[col] = np.zeros(shape=(len(output_X_test)))

    output_X_test = output_X_test[input_train_cols]

    return output_X_test

#%% Handle imbalanced datasets

def balance_data(input_X_train, input_y_train, target_ratio, approach='SMOTE'):
    valid_approaches = ['random_undersampling', 'random_oversampling', 'SMOTE']
    assert approach in valid_approaches
    
    if approach == 'random_undersampling':
        n_positive = np.sum(input_y_train)
        n_total =  int(n_positive / target_ratio)
        n_negative = n_total - n_positive
        model = RandomUnderSampler(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
    
    elif approach == 'random_oversampling':
        n_negative = np.sum(input_y_train == 0)
        n_positive = int((n_negative / (1 - target_ratio)) - n_negative)
        model = RandomOverSampler(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
        
    elif approach == 'SMOTE':
        # SMOTE -- Need to remove null values and convert categorial variables before using
        assert len(input_X_train.select_dtypes('object').columns) == 0, "Need to remove categorical features before using SMOTE"
        assert np.sum(np.sum(input_X_train.isnull())) == 0, "Need to remove null values before using SMOTE"

        n_negative = np.sum(input_y_train == 0)
        n_positive = int((n_negative / (1 - target_ratio)) - n_negative)
        model = SMOTE(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
    
    else:
        print("Please select one of: {} as balancing approach".format(valid_approaches))
    
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
    
    def fit_df(self, input_df):
        for col in input_df:
            self.fit(input_df[col])
        
        return self.model_parameters
    
    def transform_df(self, input_df):
        output_df = input_df.copy()
        for col in output_df:
            if col not in self.model_parameters:
                continue
            else:
                output_df[col] = self.transform(output_df[col])
        
        return output_df
    
    def fit_transform_df(self, input_df):
        output_df = input_df.copy()
        for col in output_df:
            self.fit(output_df[col])
            if col not in self.model_parameters:
                continue
            else:
                output_df[col] = self.transform(output_df[col])
        
        return output_df


class MeanEncoder():
    def __init__(self):
        self.model_parameters = {}
        # When the value is not in the training set then can use the target mean
        self.target_mean = 0
    
    def fit(self, input_series, target_output):
        merged_df = pd.merge(input_series, target_output, how='left', left_index=True, right_index=True)
        mean_target_value = merged_df.groupby(input_series.name).mean()
        self.model_parameters[input_series.name] = mean_target_value[target_output.name].to_dict()
        self.target_mean = np.mean(target_output)
        return self.model_parameters[input_series.name]
        
    def transform(self, input_series):
        return input_series.map(self.model_parameters[input_series.name]).fillna(self.target_mean)
    
    def fit_df(self, input_df, target_output):
        for col in input_df:
            self.fit(input_df[col], target_output)
        
        return self.model_parameters
    
    def transform_df(self, input_df):
        output_df = input_df.copy()
        for col in output_df:
            output_df[col] = self.transform(output_df[col])
        
        return output_df
    
    def fit_transform_df(self, input_df, target_output):
        output_df = input_df.copy()
        for col in output_df:
            self.fit(output_df[col], target_output)
            output_df[col] = self.transform(output_df[col])
        
        return output_df

class BinEncoder():
    def __init__(self, equal_sized_bins=True):
        self.model_parameters = {}
        self.equal_sized_bins = equal_sized_bins
        
    def fit(self, input_series, n_bins=5):
        if self.equal_sized_bins:
            # The bins have the same number of elements in each 
            _, self.model_parameters[input_series.name] = pd.qcut(input_series, q=5, retbins=True)
        
        else:
            #The bins cover an equal amount of distribution in each 
            _, self.model_parameters[input_series.name] = pd.cut(input_series, bins=5, retbins=True)
            
        return self.model_parameters[input_series.name]
    
    def transform(self, input_series):
        bins = self.model_parameters[input_series.name]
        idx = pd.cut(input_series, bins=bins, labels=np.arange(len(bins) -1))
        
        return idx
    
    def fit_df(self, input_df, n_bins=5):
        for col in input_df:
            self.fit(input_df[col], n_bins)
        
        return self.model_parameters
    
    def transform_df(self, input_df):
        output_df = input_df.copy()
        for col in output_df:
            output_df[col] = self.transform(output_df[col])
        
        return output_df
    
    def fit_transform_df(self, input_df, n_bins=5):
        output_df = input_df.copy()
        for col in output_df:
            self.fit(output_df[col], n_bins)
            output_df[col] = self.transform(output_df[col])
        
        return output_df

def one_hot_encode_column(input_series):
    one_hot_df = pd.get_dummies(input_series)
    one_hot_df.columns = [input_series.name+"_"+str(x) for x in one_hot_df.columns]
    
    return one_hot_df

def fit_cat_encoding(input_X_train, input_y_train, categorical_encodings_dict, verbose=True):
    removed_features = []

    for col_name in categorical_encodings_dict:
        if col_name not in input_X_train.columns:
            removed_features.append(col_name)
        else:        
            if categorical_encodings_dict[col_name]['encoding'] == 'mean':
                mean_encoder = MeanEncoder()
                mean_encoder.fit(input_X_train[col_name], input_y_train)
                categorical_encodings_dict[col_name]['model'] = mean_encoder

    if (len(removed_features) > 0) & (verbose):
        print("Cat encoding fitting warning: can't fit {} columns as they have been removed".format(removed_features))

    return categorical_encodings_dict

def transform_cat_encoding(input_df, categorical_encodings_dict, verbose=True):
    output_df = input_df.copy()
    removed_features = []
    
    for col_name in categorical_encodings_dict:
        if col_name not in input_df.columns:
            removed_features.append(col_name)
        else:
            if categorical_encodings_dict[col_name]['encoding'] == 'mean':
                mean_encoder = categorical_encodings_dict[col_name]['model']
                output_df[col_name] = mean_encoder.transform(output_df[col_name])
    
            elif categorical_encodings_dict[col_name]['encoding'] == 'one-hot':
                if len(set(output_df[col_name])) == 2:
                    top_value = output_df[col_name].value_counts().index[0]
                    binary_col_name = col_name + " is " + top_value
                    output_df[binary_col_name] = np.where(output_df[col_name] == top_value, 1, 0)
                
                else:
                    one_hot_columns = one_hot_encode_column(output_df[col_name])
                    output_df = pd.concat([output_df, one_hot_columns], axis=1)
                
                output_df.drop(col_name, axis=1, inplace=True)
            
            else:
                "Please select either 'mean' or 'one-hot' encoding"
    
    if (len(removed_features) > 0) & (verbose):
        print("Cat encoding transform warning: can't transform {} columns as they have been removed".format(removed_features))
    
    return output_df

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