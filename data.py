import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import os

import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

import seaborn as sns


#%% TODO: 
# - Can potentially group the loading and saving functions into one with a switch on filetype
# - Can add enhanced imputation strategies (KNN, Decision Tree, etc.)
# 



#%% Data Loading and Saving

def create_folder(input_folder_path):
    assert os.path.exists(input_folder_path) is False, "Folder already exists"

    os.makedirs(input_folder_path)

    return True

def read_json_object(input_file_path):
    """Only designed for loading json files"""
    assert '.json' in input_file_path, 'Please ensure file is .json filetype'
    with open(input_file_path) as json_file:
        results_dict = json.load(json_file)
    return results_dict

def save_json_object(output_object, output_file_path, overwrite=False):
    assert ".json" in output_file_path, "Please ensure format is saved as .json"

    if (overwrite is False) & (os.path.exists(output_file_path)):
        raise Exception("File '{}' exists. Please either set overwrite=True or rename the file.".format(
            output_file_path))
    else:
        with open(output_file_path, 'w') as json_out:
            json.dump(output_object, json_out)
        print("File saved successfuly at {}".format(output_file_path))

    return True

def read_txt_file(input_file_path):
    """Only designed for loading txt files"""
    assert ".txt" in input_file_path, "Please ensure file format is .txt"

    with open(input_file_path, encoding='utf8') as f:
        content = f.read()

    return content

def save_txt_file(output_object, output_file_path, overwrite=False):
    assert ".txt" in output_file_path, "Please ensure format is saved as .txt"

    if (overwrite is False) & (os.path.exists(output_file_path)):
        raise Exception("File '{}' exists. Please either set overwrite=True or rename the file.".format(
            output_file_path))
    else:
        with open(output_file_path, 'w', encoding='utf8') as f:
            f.write(output_object)
        print("File saved successfuly at {}".format(output_file_path))

    return True

def read_pickle_object(file_name):
    """Only designed for loading pickle files"""
    assert '.pickle' in file_name, 'Please ensure file is .pickle filetype'
    with open(file_name, 'rb') as handle:
        pickle_object = pickle.load(handle)
    return pickle_object

def save_pickle_object(output_object, output_file_path, overwrite=False):
    assert ".pickle" in output_file_path, "Please ensure format is saved as .pickle"

    if (overwrite is False) & (os.path.exists(output_file_path)):
        raise Exception("File '{}' exists. Please either set overwrite=True or rename the file.".format(
            output_file_path))
    else:
        with open(output_file_path, "wb") as pickle_out:
            pickle.dump(output_object, pickle_out)
        print("File saved successfuly at {}".format(output_file_path))

    return True




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

np.sum(np.where(DF.y_train > 350000, 1 , 0)) / len(DF.y_train)

DF.y_train = np.where(DF.y_train > 350000, 1 , 0)

np.sum(DF.y_train) / len(DF.y_train)

# Undersampling
from imblearn.under_sampling import RandomUnderSampler

desired_ratio = 0.25
n_positive = np.sum(DF.y_train)
n_total =  int(n_positive / desired_ratio)
n_negative = n_total - n_positive

rus = RandomUnderSampler(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
X_res, y_res = rus.fit_resample(DF.X_train, DF.y_train)

# Oversampling
from imblearn.over_sampling import RandomOverSampler

desired_ratio = 0.25
n_negative = np.sum(DF.y_train == 0)
n_positive = int((n_negative / (1 - desired_ratio)) - n_negative)

rus = RandomOverSampler(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
X_res, y_res = rus.fit_resample(DF.X_train, DF.y_train)

# SMOTE -- Need to remove null values and convert categorial variables before using
from imblearn.over_sampling import SMOTE
assert len(DF.X_train.select_dtypes('object')) == 0, "Need to remove categorical features before using SMOTE"
assert np.sum(np.sum(DF.X_train.isnull())) ==0, "Need to remove null values before using SMOTE"

desired_ratio = 0.25
n_negative = np.sum(DF.y_train == 0)
n_positive = int((n_negative / (1 - desired_ratio)) - n_negative)
sm = SMOTE(random_state=42, sampling_strategy = {0 : n_negative, 1 : n_positive})
X_res, y_res = sm.fit_resample(DF.X_train, DF.y_train)


#%%


si = SimpleImputer()

[si.fit(DF.X_train[col], 'mode') for col in DF.X_train.columns]

for col in DF.X_train.columns:
    DF.X_train.loc[:, col] = si.transform(DF.X_train[col])


#%%
df = DF.X_train
prop_column_null(df)[:20]

input_series = DF.X_train.LotFrontage

input_series.mean()

si = SimpleImputer()

si.fit(DF.X_train.LotFrontage, 'mode')

si.transform(DF.X_train.LotFrontage)

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





#%%

input_df = pd.read_csv('data/train.csv')
target_name = 'SalePrice'
split_ratios = {'train' : 0.60,
                'validation' : 0.20,
                'test' : 0.20}

DF = ModelData(input_df, target_name, split_ratios)


sc = StandardScaler()
me = MeanEncoder()
bin = BinEncoder()

#%%

me.fit(DF.X_train.MSZoning, DF.y_train)
me.fit(DF.X_train.SaleType, DF.y_train)

me.model_parameters

a = mean_encode_column(DF.X_train.MSZoning, DF.y_train)

b = me.transform(DF.X_train.MSZoning)

bin.fit(DF.X_train.Id)

bin.transform(DF.X_train.Id)

#%%







# Does not require state for test transform



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


#%%

import numpy as np
import matplotlib.pyplot as plt

#make this example reproducible
np.random.seed(0)

#create beta distributed random variable with 200 values
data = DF.X_train.LotFrontage

#create log-transformed data
data_log = log_transform_column(data)

#define grid of plots
fig, axs = plt.subplots(nrows=1, ncols=2)

#create histograms
axs[0].hist(data, edgecolor='black')
axs[1].hist(data_log, edgecolor='black')

#add title to each histogram
axs[0].set_title('Original Data')
axs[1].set_title('Log-Transformed Data')







#%% Visualize categorical features

# Create function for charting bar chart 
def categorical_bar_chart(labels, vals, title, width=0.8):
    n = len(vals)
    _labels = np.arange(len(labels))
    for i in range(n):
        plt.bar(_labels - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge")   
    plt.xticks(_labels, labels, rotation='vertical')
    
    plt.title(title)
    plt.ylabel('Proportion of Total')
    plt.legend(('True', 'False'))

# Create function for charting scatter plot
def scatter_plot(array1_norm, array2_norm, title):
    plt.title(title)
    plt.scatter(array1_norm,array2_norm);
    line = np.linspace(0.0,max(array1_norm)+.01,10)
    plt.plot(line,line, '--r');
    plt.ylabel('Proportion True')
    plt.xlabel('Proportion False')

    for i, txt in enumerate(array1_norm.index):
        plt.annotate(txt, (array1_norm.iloc[i], array2_norm.iloc[i]))

# Calculate the differences between datasets for a given feature
# Are focusing on the differences between the means of the various features
    # and want to focus on only the most extreme instances
# Given some features have a high number of unique values (i.e. borough) function accepts
    # a maximum number of elements

def calculate_differences(feature, chart, max_elements = 10):    
    array1 = positive_df[feature]
    array2 = negative_df[feature]
    
    array1_norm = array1.value_counts()/len(array1)
    array2_norm = array2.value_counts()/len(array2)
    
    # Ensure arrays have the same elements for comparison
    shared_elements = array1_norm.index.intersection(array1_norm.index)
    array1_norm = array1_norm.loc[shared_elements]
    array2_norm = array2_norm.loc[shared_elements]
    
    assert len(array1_norm) == len(array2_norm), 'Arrays are not equal'
    
    # In cases with more groupings than max_elements,include only most extreme elements
    delta = (array1_norm) - (array2_norm)
    delta.sort_values(ascending=False, inplace=True)

    if len(array1_norm) > max_elements: 
        largest_difference = pd.concat([delta[:min(len(array1),int(max_elements/2))],
                                        delta[-min(len(array1),int(max_elements/2)):]])
    else:
        largest_difference = delta
        
    # Select which chart output is preferred (between scatter and bar)
    if chart.lower() == 'bar':
        difference_array = [array1_norm.loc[largest_difference.index], array2_norm.loc[largest_difference.index]]
        return categorical_bar_chart(largest_difference.index,difference_array,feature)
    
    else:
        return scatter_plot(array1_norm.loc[largest_difference.index], array2_norm.loc[largest_difference.index], feature)

# Create layout for multiple charts and plot using for loop
def examine_features(features_list):
    columns = 2
    rows = len(features_list)
    fig = plt.figure(figsize=(12, 6*rows))
    j = 0

    for i in range(rows*columns):
        fig.add_subplot(rows, columns, i+1)

        if i % 2 == 0:
            calculate_differences(features_list[j], 'scatter')
        else:
            calculate_differences(features_list[j], 'bar')
            j += 1

    plt.tight_layout()
    plt.show()

#%% Plot Correlation and Pair Plot 

# Pair Plot
def pair_plot(input_df):
    plt.figure(figsize=(10,8), dpi= 80)
    sns.pairplot(input_df, kind="reg", hue="target")
    plt.show()

def plot_correlation_map(input_df):
    # Create correlation heatmap of important features and target
    sns.set(style="white")
    
    # Compute the correlation matrix
    corr = input_df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 7))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=False, linewidths=.5, annot=True, fmt=".2f", cbar=False, yticklabels = True, xticklabels = False);


#%% Handle categorical features

def one_hot_encode(X_train_input, X_test_input):
    X_train_output = pd.get_dummies(X_train_input)
    X_train_output.columns = [X_train_input.name+': '+x for x in X_train_output.columns]
    
    X_test_output = pd.get_dummies(X_test_input)
    X_test_output.columns = [X_test_input.name+': '+x for x in X_test_output.columns]
    return X_train_output, X_test_output


def mean_encode(X_train_input, X_test_input):
    mean_encodings = df[[X_train_input.name,'target']].loc[X_train_input.index].groupby(X_train_input.name).mean()
    X_train_output = X_train_input.to_frame().merge(mean_encodings,how='left',on=X_train_input.name)
    X_test_output = X_test_input.to_frame().merge(mean_encodings,how='left',on=X_test_input.name)
    return X_train_output['target'].values, X_test_output['target'].values
    

def encode_cat_vars(X_train_input, X_test_input, cat_encoding_dict):
    for feature, encoding in cat_encoding_dict.items():
        if encoding == 'one-hot':
            X_train_output, X_test_output = one_hot_encode(X_train_input[feature], X_test_input[feature])
            X_train_input = X_train_input.join(X_train_output)
            X_test_input = X_test_input.join(X_test_output)

            X_train_input.drop(feature,axis=1,inplace=True)
            X_test_input.drop(feature,axis=1,inplace=True)

        elif encoding == 'mean-encode':
            X_train_input[feature], X_test_input[feature] = mean_encode(X_train_input[feature], X_test_input[feature])
            
        # Ensure all columns are shared between train / test, if not then add zero column to test set
        for col in X_train_input.columns:
            if col not in X_test_input.columns:
                X_test_input[col] = [0] * len(X_test_input)  
    return X_train_input, X_test_input
    

#%% Train / Test split & Standardize
def standardize(X_train_input, X_test_input):
    std_cols = X_train_input.columns[~((X_train_input==0) | (X_train_input==1)).all()]

    #Standardize features which require it in 'std_cols'
    sc = StandardScaler()

    std_train_df = X_train_input.loc[:,std_cols].copy()
    std_test_df = X_test_input.loc[:,std_cols].copy()

    sc.fit(std_train_df)

    X_train_input.loc[:,std_cols] = sc.transform(std_train_df)
    X_test_input.loc[:,std_cols] = sc.transform(std_test_df)


def get_train_test_splits(input_df, target_name='target', test_size=0.30):
    X_features = input_df.loc[:, input_df.columns != target_name]
    y = input_df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=34)
    return X_train, X_test, y_train, y_test 


def export_df_to_csv(X_input, y_input, file_name):
    X_input['target'] = y_input
    X_input.to_csv(file_name, index=False)

# Implement SMOTE


#%% Load Data and remove null values
#p = 0.01  # 1% of the lines
## keep the header, then take only 1% of lines
## if random from [0,1] interval is greater than 0.01 the row will be skipped
#df = pd.read_csv(
#         'trans.asc', sep=';',
#         header=0, 
#         skiprows=lambda i: i>0 and random.random() > p
#)

df = pd.read_csv('data/titanic.csv') #head=None

# Check for null values
df.info()

null_rows = df.loc[df.isnull().sum(axis=1) >= 1].copy()

# Fill null values
    # Can either: 1) drop values 2) Fill with mean 3) use simple model to fill

# Drop values
#df.dropna(axis=0, inplace=True)

# Drop columns
df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True)
df.dropna(subset=['Embarked'], axis=0, inplace=True)

# Fill full df with mean                # Fill single column: df['proline'] = df.proline.fillna(df.proline.mean())
df.fillna(df.mean(), inplace=True)

df.rename(columns={'Survived':'target'}, inplace=True)

#%% Visualize data

# Visualize target distribution
df.target.value_counts().plot(kind='bar');
df.target.value_counts(), df.target.value_counts()/ len(df.target)

# Check for outliers
plt.hist(df.SibSp)
df.describe()
np.percentile(df.SibSp, 99)

df = df.loc[df.Parch < 3]

# Examine categorical features
features = ['Pclass', 'Sex', 'Embarked']
positive_df = df[df['target']==1]
negative_df = df[df['target']==0]

examine_features(features)

# Examine feature correlation
plot_correlation_map(df)

#%% Feature engineering

# Map single binary feature
df['Sex'] = df.Sex.map({'male':1, 'female': 0})

# Bin certain continuous features
#bin_feats(df, ['Age', 'Fare'], 5)

# Split into train / test split
X_train, X_test, y_train, y_test = get_train_test_splits(df, 'target', test_size=0.30)

# Encode categorical variables
cat_encoding_dict = {'Embarked':'mean-encode'}
X_train, X_test = encode_cat_vars(
        X_train, X_test, cat_encoding_dict=cat_encoding_dict)

X_train_std, X_test_std = X_train.copy(), X_test.copy()

# Standardize features
standardize(X_train_std, X_test_std)

#%%
export_df_to_csv(X_train, y_train, 'data/train_data.csv')
export_df_to_csv(X_test, y_test, 'data/test_data.csv')

export_df_to_csv(X_train_std, y_train, 'data/std_train_data.csv')
export_df_to_csv(X_test_std, y_test, 'data/std_test_data.csv')

#%% ------------------------- Archive ----------------------------

# Requires state for test transform
# def standardize(X_train_input, X_test_input):
#     X_train_input = X_train_input.select_dtypes(exclude=['object'])
#     std_cols = X_train_input.columns[~((X_train_input==0) | (X_train_input==1)).all()]

#     #Standardize features which require it in 'std_cols'
#     sc = StandardScaler()

#     std_train_df = X_train_input.loc[:,std_cols].copy()
#     std_test_df = X_test_input.loc[:,std_cols].copy()

#     sc.fit(std_train_df)

#     results = {'std_columns' : std_cols,
#                'std_scalar_model' : sc,
#                'X_train_std' : pd.DataFrame(sc.transform(std_train_df), columns=std_cols),
#                'X_test_std' : pd.DataFrame(sc.transform(std_test_df), columns=std_cols)}
    
#     return results 

# Handle categorical features
# def standardize_column(input_series):
#     series_mean = input_series.mean()
#     series_std_dev = input_series.std()
    
#     standardized_series = (input_series - series_mean) / series_std_dev
    
#     return {'transformed_series' : standardized_series,
#             'model': {'series_mean' : series_mean, 'series_std_dev' : series_std_dev}}

# def mean_encode_column(input_series, target_output):
#     assert len(input_series) == len(target_output), "Array lengths are not equal"
#     merged_df = pd.merge(input_series, target_output, how='left', left_index=True, right_index=True)
#     mean_target_value = merged_df.groupby(input_series.name).mean()
#     mean_target_dict = mean_target_value[target_output.name].to_dict()
    
#     return {'transformed_series' : input_series.map(mean_target_dict),
#             'model' : mean_target_dict }

# # Bucketize continuous variables
# def bin_column(input_series, n_bins=5):
#     """bins column into n equally spaced bins (do NOT have equal number of rows)"""
#     sep = (input_series.size/float(n_bins)) * np.arange(1, n_bins+1)
#     idx = sep.searchsorted(np.arange(input_series.size))
    
#     return idx[input_series.argsort().argsort()]