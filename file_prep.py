#%% Import dependencies
import os
import pandas as pd
import numpy as np
import pickle
import json
import chardet
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import random

# %matplotlib inline
sns.set_style("white")


#%%

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
            print("Used pd.read_csv")
            return content
            
        except Exception as e:
            print(e)
            with open(file, 'rb') as rawdata:
                result = chardet.detect(rawdata.read(10000))
            
                # Note: low memory option will enable columns with mixed data types to be asserted later
                content = pd.read_csv(file, encoding=result['encoding'], low_memory=False)
                print("Used chardet override")
                
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
            output_object.to_csv(output_file_path, index=False)
            
        else:
            print("Please select one of: {} file types".format(accepted_file_types))
        
        print("File saved successfuly at {}".format(output_file_path))

def sample_data(input_df, fraction=0.40):
    return pd.sample(frac=fraction, replace=False, random_state=34)




#%% Handle Null Values

def get_prop_null_column(input_df):
    """Returns the proportion of null values by column"""
    return (input_df.isnull().sum(axis = 0) / len(input_df)).sort_values(ascending=False)

def get_prop_null_row(input_df):
    """Returns the proportion of null values by row"""
    return (input_df.isnull().sum(axis = 1) / len(input_df.columns)).sort_values(ascending=False)


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

#%% Outlier detection

def is_IQR_outlier(input_series):
    assert input_series.dtype != 'O', "Cannot use inter-quartile range for strings"
    q75, q25 = np.percentile(input_series, [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - (1.5 * iqr)
    upper_bound = q75 + (1.5 * iqr)
    
    return (input_series > upper_bound) | (input_series < lower_bound)

def get_IQR_outliers(input_X_train, n_threshold=2):
    results = {}
    IQR_outliers = input_X_train.apply(is_IQR_outlier, axis=0)
    results['n_outliers_by_col'] = IQR_outliers.sum(axis=0).sort_values(ascending=False)
    results['n_outliers_by_row'] = IQR_outliers.sum(axis=1)
    results['outlier_rows'] = input_X_train.loc[results['n_outliers_by_row'] > n_threshold]
    results['outlier_proportion'] = np.around(len(results['outlier_rows']) / len(input_X_train), 2)
    return results

def get_isolation_forest_outliers(input_X_train, est_outlier_prop='auto'):
    results = {}
    clf = IsolationForest(
        contamination=est_outlier_prop, random_state=34).fit(input_X_train)
    isol_forest_outliers = clf.predict(input_X_train)
    results['outlier_rows'] = input_X_train.loc[isol_forest_outliers == -1]
    results['outlier_proportion'] = np.around(len(results['outlier_rows']) / len(input_X_train), 2)
    return results

def get_local_outlier_factor_outliers(input_X_train, n_neighbors=20, est_outlier_prop='auto'):
    results = {}
    lof_outliers = LocalOutlierFactor(
        n_neighbors=n_neighbors, 
        contamination=est_outlier_prop).fit_predict(input_X_train)
    results['outlier_rows'] = input_X_train.loc[lof_outliers == -1]
    results['outlier_proportion'] = np.around(len(results['outlier_rows']) / len(input_X_train), 2)
    return results

#%% Data profiling

def create_html_data_profile(input_df, output_file_path, overwrite=False):
    _, file_extension = os.path.splitext(output_file_path)
    if file_extension != '.html':
        output_file_path+'.html'
        
    if (overwrite is False) & (os.path.exists(output_file_path)):
        raise Exception("File '{}' exists. Please either set overwrite=True or rename the file.".format(
            output_file_path))
    
    
    profile = ProfileReport(input_df,
                            minimal=True)

    profile.to_file(output_file_path)
    
    return True


#%% Plot Correlation and Pair Plot 

def pair_plot(input_X_train, input_y_train):
    input_df = pd.concat([input_X_train, input_y_train], axis=1)
    sns.pairplot(
        input_df, 
        diag_kind='hist', 
        plot_kws=dict(s=14, alpha=.2, linewidth=0))
    plt.show()

def plot_correlation_map(input_X_train, input_y_train):
    input_df = pd.concat([input_X_train, input_y_train], axis=1)   
    fig, ax = plt.subplots(figsize = [14,12])
    colormap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(input_df.corr(),
                cmap = colormap,
                center = 0,
                annot = True,
                linewidths = 0.1,
                fmt=".2f")

#%% # Calculate the differences between datasets for a given categorial feature
    # Are focusing on the differences between the means of the various features
        # and want to focus on only the most extreme instances
    # Given some features have a high number of unique values, function accepts
        # a maximum number of elements

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

def calculate_differences(input_X_train, input_y_train, feature_list, max_compare_elements=10):
    positive_df = input_X_train.loc[input_y_train == 1]
    negative_df = input_X_train.loc[input_y_train == 0]
    
    for feature in feature_list:
        positive_series = positive_df[feature]
        negative_series = negative_df[feature]
         
        positive_series_norm = positive_series.value_counts()/len(positive_series)
        negative_series_norm = negative_series.value_counts()/len(negative_series)
     
        # Add elements if not contained in series
        elements_only_in_positive = list(set(positive_series_norm.index) - set(negative_series_norm.index))
        elements_only_in_negative = list(set(negative_series_norm.index) - set(positive_series_norm.index))
     
        for element in elements_only_in_positive:
            negative_series_norm[element] = 0
            
        for element in elements_only_in_negative:
            positive_series_norm[element] = 0
             
        assert len(positive_series_norm) == len(negative_series_norm), 'Arrays are not equal'
        
        # In cases with more groupings than max_elements,include only most extreme elements
        delta = (positive_series_norm) - (negative_series_norm)
        delta.sort_values(ascending=False, inplace=True)
    
        if len(positive_series_norm) > max_compare_elements: 
            largest_difference = pd.concat([delta[:min(len(positive_series), int(max_compare_elements/2))],
                                            delta[-min(len(positive_series), int(max_compare_elements/2)):]])
        else:
            largest_difference = delta
            
        difference_array = [positive_series_norm.loc[largest_difference.index], negative_series_norm.loc[largest_difference.index]]
        categorical_bar_chart(largest_difference.index, difference_array, feature)
    
        plt.show()
    
    return True

#%%

# articles_df = pd.read_csv('D:/HM_Data/articles.csv')

df = load_file('D:/HM_Data/articles.csv')

# Load 10% of transactions to make it easy to work with
p = 0.02  
transactions_df = pd.read_csv(
          'D:/HM_Data/transactions_train.csv',
          header=0, 
          skiprows=lambda i: i>0 and random.random() > p
)

#%%

# Explore sample of records 
sample_transactions_df = transactions_df.sample(10)
sample_df = df.sample(10)
df.info()
df.describe().T

b = df.loc[df['product_code'] == 783707]

# Count by
df.groupby(['colour_group_code', 'colour_group_name']).article_id.nunique().sort_values(ascending=False)

# Create profile of data
# create_html_data_profile(df, 'outputs/articles_profile.html')
# create_html_data_profile(transactions_df, 'outputs/transactions_profile.html')

#%%

# Check for duplicates
cols_excluding_pk = df.drop('article_id', axis=1).columns

# Inspect Duplicates
c = get_duplicate_rows(df, column_subset=cols_excluding_pk)
c = df.loc[(df['product_code'] == 118458) &  (df['graphical_appearance_no'] == 1010010)]

# Drop duplicates
df = drop_duplicate_rows(df, column_subset=cols_excluding_pk)

# Drop columns 
df.drop(['product_code', 'product_type_no', 'colour_group_code'], axis=1, inplace=True)


#%% Inspect Null Values

null_col_prop = get_prop_null_column(df)
null_row_prop = get_prop_null_row(df)

# Drop null rows over threshold
df.drop(
        null_row_prop.loc[null_row_prop > 0.04].index, axis=0, inplace=True)
df.reset_index(drop=True)

#%% Identify outliers


IQR_outliers = transactions_df.loc[is_IQR_outlier(transactions_df['price'])]

# isolation_forest_outliers = get_isolation_forest_outliers(pd.DataFrame(transactions_df['price']), est_outlier_prop='auto')

transactions_df['is_outlier'] = is_IQR_outlier(transactions_df['price'])


#%% Export processed data

save_file(df, 'outputs/articles_df.csv')

transactions_df = transactions_df.loc[transactions_df['article_id'].isin(df['article_id'])]

save_file(transactions_df, 'outputs/transactions_df.csv', overwrite=True)


#%% Confirmatory Analysis

df = pd.merge(transactions_df, df, how='left', on='article_id')

#%%

df.groupby('product_type_name').article_id.count().sort_values(ascending=False)

x = df.loc[pd.isnull(df['product_type_name'])]

#%%







