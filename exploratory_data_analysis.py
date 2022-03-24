import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm, uniform, expon
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import os

# %matplotlib inline
sns.set_style("white")

#%% Profile data

def prop_column_null(input_df):
    """Returns the proportion of null values by column"""
    return (input_df.isnull().sum(axis = 0) / len(input_df)).sort_values(ascending=False)

def prop_row_null(input_df):
    """Returns the proportion of null values by row"""
    return (input_df.isnull().sum(axis = 1) / len(input_df.columns)).sort_values(ascending=False)


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

#%% Quantile-Quantile plot 

def get_qq_plot(input_series, comparison_distribution='normal'):
    valid_comparison_distributions = ['normal', 'uniform', 'exponential']
    if comparison_distribution == 'normal':
        sm.qqplot(input_series, norm, fit=True, line="45")
        
    elif comparison_distribution == 'uniform':
        sm.qqplot(input_series, uniform, fit=True, line="45")
    
    elif comparison_distribution == 'exponential':
        sm.qqplot(input_series, expon, fit=True, line="45")
        
    else:
        print("Please select one of: {} as comparison distribution".format(
            valid_comparison_distributions))

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

