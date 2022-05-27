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
import operator as op

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
    output_df = output_df.drop(duplicate_rows.index)
    output_df = output_df.reset_index(drop=True)
    
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

#%% Cohort Analysis

def get_cohorts(df, period='M'):
    """Given a Pandas DataFrame of transactional items, this function returns
    a Pandas DataFrame containing the acquisition cohort and order cohort which
    can be used for customer analysis or the creation of a cohort analysis matrix.
    
    Parameters
    ----------

    df: Pandas DataFrame
        Required columns: order_id, customer_id, order_date
    period: Period value - M, Q, or Y
        Create cohorts using month, quarter, or year of acquisition
    
    Returns
    -------
    products: Pandas DataFrame
        customer_id, order_id, order_date, acquisition_cohort, order_cohort
    """
    
    df = df[['customer_id','order_id','order_date']].drop_duplicates()
    df = df.assign(acquisition_cohort = df.groupby('customer_id')\
                   ['order_date'].transform('min').dt.to_period(period))
    df = df.assign(order_cohort = df['order_date'].dt.to_period(period))
    return df



def get_retention(df, period='M'):
    """Calculate the retention of customers in each month after their acquisition 
    and return the count of customers in each month. 
    
    Parameters
    ----------

    df: Pandas DataFrame
        Required columns: order_id, customer_id, order_date
    period: Period value - M, Q, or Y
        Create cohorts using month, quarter, or year of acquisition
    
    Returns
    -------
    products: Pandas DataFrame
        acquisition_cohort, order_cohort, customers, periods
    """

    df = get_cohorts(df, period).groupby(['acquisition_cohort', 'order_cohort'])\
                                .agg(customers=('customer_id', 'nunique')) \
                                .reset_index(drop=False)
    df['periods'] = (df.order_cohort - df.acquisition_cohort).apply(op.attrgetter('n'))

    return df

def get_cohort_matrix(df, period='M', percentage=False):
    """Return a cohort matrix showing for each acquisition cohort, the number of 
    customers who purchased in each period after their acqusition. 
    
    Parameters
    ----------

    df: Pandas DataFrame
        Required columns: order_id, customer_id, order_date
    period: Period value - M, Q, or Y
        Create cohorts using month, quarter, or year of acquisition
    percentage: True or False
        Return raw numbers or a percentage retention

    Returns
    -------
    products: Pandas DataFrame
        acquisition_cohort, period
    """
    
    df = get_retention(df, period).pivot_table(index = 'acquisition_cohort',
                                               columns = 'periods',
                                               values = 'customers')
    
    if percentage:
        df = df.divide(df.iloc[:,0], axis=0)*100
    
    return df

#%% Load Data

df = pd.read_csv(
    'C:/Users/andre/OneDrive/Documents/Projects/insights/outputs/transaction_items.csv')

df.rename({
    'InvoiceNo' : 'order_id',
    'StockCode': 'sku',
	'Description': 'description',
	'Quantity' : 'quantity',
	'InvoiceDate' : 'order_date',
 	'UnitPrice' : 'unit_price',
     'CustomerID' : 'customer_id',
     'Country': 'country'}, axis=1, inplace=True)

df['order_date'] = pd.to_datetime(df['order_date'])

#%%

# Load 10% of transactions to make it easy to work with
# p = 0.001  
# transactions_df = pd.read_csv(
#           'D:/HM_Data/transactions_train.csv',
#           header=0, 
#           skiprows=lambda i: i>0 and random.random() > p
# )

# articles_df = pd.read_csv('D:/HM_Data/articles.csv')
# df = load_file('D:/HM_Data/articles.csv')

#%% Exploratory Analysis

# Explore sample of records 
sample_transactions_df = df.sample(10)
sample_df = df.sample(10)
df.info()
df.describe().T

# Count by
# df.groupby(['customer_id', 'sku']).article_id.nunique().sort_values(ascending=False)

# Create profile of data
# create_html_data_profile(df, 'outputs/df_profile.html')

#%% Preprocessing

# Check for duplicates
# cols_excluding_pk = df.drop('article_id', axis=1).columns

# Inspect Duplicates
duplicates = get_duplicate_rows(df) #get_duplicate_rows(df, column_subset=cols_excluding_pk)
# df.loc[(df['product_code'] == 118458) &  (df['graphical_appearance_no'] == 1010010)]

# Drop duplicates
df = drop_duplicate_rows(df) #drop_duplicate_rows(df, column_subset=cols_excluding_pk)

# Drop columns 
# df.drop(['product_code', 'product_type_no', 'colour_group_code'], axis=1, inplace=True)


#%% Inspect Null Values

null_col_prop = get_prop_null_column(df)
null_row_prop = get_prop_null_row(df)

# Fill Null values
df['customer_id'] = df['customer_id'].fillna('No_ID')

# Drop null rows over threshold
row_drop_threshold = 0.50

df.drop(
        null_row_prop.loc[null_row_prop > row_drop_threshold].index, axis=0, inplace=True)
df.reset_index(drop=True)

#%% Change Types

df['customer_id'] = df['customer_id'].astype(str)


#%% Identify outliers

# IQR_outliers = df.loc[is_IQR_outlier(df['price'])]

# isolation_forest_outliers = get_isolation_forest_outliers(pd.DataFrame(df['price']), est_outlier_prop='auto')

# transactions_df['is_outlier'] = is_IQR_outlier(df['price'])


#%% Preprocessing

# Remove negative quantities and negative unit prices
df = df.loc[(df['quantity'] > 0) & (df['unit_price'] > 0), :]
df.reset_index(drop=True, inplace=True)

# Calculate total revenue
df['total_revenue'] = df['quantity'] * df['unit_price']
df['year'] = df['order_date'].dt.year

#%% Export processed data

# save_file(df, 'outputs/transactions_df.csv', overwrite=True)



#%% Build customer history summary

# Columns are: ['order_id', 'sku', 'description', 'quantity', 'order_date', 'unit_price', 'customer_id']

# Create summary of customer history 
customer_hist_df = df[['customer_id', 'year', 'sku', 'quantity', 'unit_price', 'total_revenue']].groupby(
    ['customer_id', 'year', 'sku']).aggregate(['sum', 'mean'])

customer_hist_df.columns = ["_".join(a) for a in customer_hist_df.columns.to_flat_index()]
customer_hist_df.reset_index(inplace=True)

# Sort values
customer_hist_df.sort_values(['customer_id', 'year', 'sku'], ascending=True, inplace=True)

# Get first customer year
cust_first_year = customer_hist_df[
    ['customer_id', 'year']].groupby(['customer_id']).head(1)
cust_first_year.rename({'year' : 'cust_first_year'}, axis=1, inplace=True)

# Get first product year
prod_first_year = customer_hist_df[
    ['customer_id', 'year', 'sku']].groupby(['customer_id', 'sku']).head(1)
prod_first_year.rename({'year' : 'prod_first_year'}, axis=1, inplace=True)

# Merge customer and product first year with main dataframe
combined_df = pd.merge(
    df, cust_first_year, how='left', on='customer_id')

combined_df = pd.merge(
    combined_df, prod_first_year, how='left', on=['customer_id', 'sku'])


#%% Add Churn Data

# Get final year of product revenue
prod_final_year_rev = customer_hist_df[
    ['customer_id', 'sku', 'total_revenue_sum']].groupby(['customer_id', 'sku']).tail(1)
prod_final_year_rev.rename({'total_revenue_sum' : 'prod_final_year_revenue'}, axis=1, inplace=True)
# Make negative as churned rev
prod_final_year_rev['prod_final_year_revenue'] = -prod_final_year_rev['prod_final_year_revenue']

# Get product churn date (one year after last purchase)
prod_churn_date = df[['customer_id', 'sku', 'order_date']].groupby(
        ['customer_id', 'sku']).aggregate(['max'])

prod_churn_date.columns = ["_".join(a) for a in prod_churn_date.columns.to_flat_index()]
prod_churn_date.reset_index(inplace=True)
prod_churn_date.rename({'order_date_max' : 'prod_churn_date'}, axis=1, inplace=True)
prod_churn_date['prod_churn_date'] = prod_churn_date['prod_churn_date'] + pd.DateOffset(years=1)

# Join product churn date
churn_df = pd.merge(prod_final_year_rev, prod_churn_date, how='left', on=['customer_id', 'sku'])
churn_df.rename({'prod_churn_date' : 'order_date',
                 'prod_final_year_revenue' : 'total_revenue'}, axis=1, inplace=True)
churn_df['order_id'] = 'churned_order'

# Concat with main dataframe
final_df = pd.concat([combined_df, churn_df])
final_df.reset_index(drop=True, inplace=True)

# Get customer churn date (last product churn date)
cust_churn_date = prod_churn_date[['customer_id', 'prod_churn_date']].groupby(['customer_id']).max()
cust_churn_date.reset_index(inplace=True)
cust_churn_date.rename({'prod_churn_date' : 'cust_churn_date'}, axis=1, inplace=True)

final_df = pd.merge(final_df, cust_churn_date, how='left', on=['customer_id'])
final_df['tenure'] = final_df['cust_churn_date'].dt.year - final_df['cust_first_year']


#%% Add transaction type based on customer behaviour (customer add, product add, product churn)

final_df['type'] = str('recurring')

final_df.loc[final_df['year'] == final_df['cust_first_year'], 'type'] = 'customer_add'
final_df.loc[
    (final_df['year'] != final_df['cust_first_year']) &
    (final_df['year'] == final_df['prod_first_year']), 'type'] = 'product_add'
final_df.loc[
    (final_df['order_id'] == 'churned_order') &
    (final_df['total_revenue'] < 0), 'type'] = 'product_churn'

#%% 

save_file(final_df, 'outputs/transaction_bridge.csv', overwrite=True)


#%% Plot Cohort Analysis

# Requires the following cols: order_id, customer_id, order_date
df_matrixm = get_cohort_matrix(df[['order_id', 'customer_id', 'order_date']],
                               'M', percentage=True)
df_matrixm = df_matrixm.drop(0, axis=1)

f, ax = plt.subplots(figsize=(20, 5))
cmap = sns.color_palette("Blues")
monthly_sessions = sns.heatmap(df_matrixm, 
                    annot=True, 
                    linewidths=3, 
                    ax=ax, 
                    cmap=cmap, 
                    square=False)

ax.axes.set_title("Cohort analysis",fontsize=20)
ax.set_xlabel("Acquisition cohort",fontsize=15)
ax.set_ylabel("Period",fontsize=15)
plt.show()

#%% --------------- ARCHIVE ---------------


x = df.loc[df['customer_id'] == '13296.0']
y = final_df.loc[final_df['customer_id'] == '16446.0']


df.loc[(df['order_date'].dt.year == 2011) &
                    (df['order_date'].dt.month == 12)]['total_revenue'].sum()

final_df.loc[(final_df['order_date'].dt.year == 2011) &
                    (final_df['order_date'].dt.month == 12)]['total_revenue'].sum()


z = final_df.loc[(final_df['order_date'].dt.year == 2011) &
                    (final_df['order_date'].dt.month == 12) &
                    (final_df['type'] == 'customer_add')]

z = final_df.loc[(final_df['order_date'].dt.year == 2012) &
                    (final_df['order_date'].dt.month == 12) &
                    (final_df['type'] == 'product_churn')]

#%%





combined_df.loc[combined_df['customer_id'] == 12346.0].T



zz = z[['customer_id', 'total_revenue']].groupby('customer_id').sum()

zz = z[['sku', 'total_revenue']].groupby('sku').sum()
zz.reset_index(inplace=True)


z = combined_df.loc[(combined_df['status'] == 'product_churn')]

z[['sku', 'total_revenue']].groupby(['sku']).sum().sort_values(by='total_revenue')


#%%

churn_year = customer_hist_df[['customer_id', 'year']].groupby('customer_id').max() + 1 
churn_year.rename({'year' : 'cust_churn_year'}, axis=1, inplace=True)


customer_hist_df = pd.merge(customer_hist_df, churn_year, how='left', on='customer_id')
prod_churn_year = customer_hist_df[['customer_id', 'year', 'sku']].groupby(['customer_id', 'sku']).max() + 1 
prod_churn_year.rename({'year' : 'prod_churn_year'}, axis=1, inplace=True)

customer_hist_df = pd.merge(customer_hist_df, prod_churn_year, how='left', on=['customer_id', 'sku'])

customer_hist_df['customer_tenure'] = customer_hist_df['cust_churn_year'] - customer_hist_df['cust_first_year']
customer_hist_df['product_tenure'] = customer_hist_df['prod_churn_year'] - customer_hist_df['prod_first_year']

#%% Confirmatory Analysis

df = raw_df.copy()
# raw_df = df.copy()
df = pd.merge(transactions_df, df, how='left', on='article_id')

#%%

# df.groupby('product_type_name').article_id.count().sort_values(ascending=False)

# x = df.loc[pd.isnull(df['product_type_name'])]

#%%

df['t_dat'] = pd.to_datetime(df['t_dat'])

df['year'] = pd.to_datetime(df['t_dat']).dt.year
df['last_year'] = df['year'] - 1

#%%

annual_revenue = df[['year', 'price']].groupby('year').sum()

calendar_scaffold = pd.DataFrame({'calendar_year':[2017, 2018, 2019, 2020]})

first_tran_date = df[['customer_id', 't_dat']].groupby('customer_id').min()
first_tran_date['t_dat'] = first_tran_date['t_dat'].dt.year
first_tran_date.rename(columns={'t_dat':'add_year'}, inplace=True)
df = pd.merge(df, first_tran_date, on='customer_id', how='left')

last_tran_date = df[['customer_id', 't_dat']].groupby('customer_id').max()
last_tran_date.rename(columns={'t_dat':'last_date'}, inplace=True)
df = pd.merge(df, last_tran_date, on='customer_id', how='left')

tenure = df['last_date'] - df['first_date']

df.loc[df['add_year'] == df['t_dat'].dt.year][['add_year', 'price']].groupby('add_year').sum()

#%%

x = df[['customer_id', 'year', 'article_id']].groupby(['customer_id', 'year']).count()
x.reset_index(inplace=True)
x

repeat_customers = x['customer_id'].value_counts().index[x['customer_id'].value_counts() > 1]

df[['year','price']].loc[df['customer_id'].isin(repeat_customers)].groupby('year').sum()

customers_with_2018_t = df.loc[df['']]

#%%

df['add_year'] = df['first_date'].dt.year
df['drop_year'] = df['last_date'].dt.year + 1 

customer_adds = df[['add_year', 'price']].groupby('add_year').sum()


customers_by_year = df[['year', 'customer_id','article_id']].groupby(['year', 'customer_id']).count()
customers_by_year.rename({'article_id' : 'n_purchases'}, axis=1, inplace=True)
customers_by_year.reset_index(inplace=True)

y = customers_by_year.pivot(index='customer_id', columns='year')

y.reset_index(inplace=True)

repeat_customers = y[(y.loc[:, ('n_purchases', 2018)] > 0) & (y.loc[:, ('n_purchases', 2019)] > 0)].index





pd.merge(calendar_scaffold, customers_by_year,  how='left',
         left_on=['last_year', 'customer_id'],
         right_on=['year', 'customer_id'])


#%%

py_customer = pd.merge(df, customers_by_year,  how='left',
         left_on=['last_year', 'customer_id'],
         right_on=['year', 'customer_id'])

py_customer['customer_in_py'] = np.where(py_customer['n_purchases']>0, True, False)


top_customer = 'ffa034ef641a6daff781539adb7830bcde0745ddb2bced1afee78dfe87419c53'

x = py_customer.loc[py_customer['customer_id'] == top_customer]

# Join in customer first year
first_year = customer_hist_df[['customer_id', 'year']].groupby('customer_id').min()
first_year.rename({'year' : 'cust_first_year'}, axis=1, inplace=True)
customer_hist_df = pd.merge(customer_hist_df, first_year, how='left', on='customer_id')

# Join in product first year
prod_first_year = customer_hist_df[['customer_id', 'year', 'sku']].groupby(['customer_id', 'sku']).min()
prod_first_year.rename({'year' : 'prod_first_year'}, axis=1, inplace=True)
customer_hist_df = pd.merge(customer_hist_df, prod_first_year, how='left', on=['customer_id', 'sku'])



# Merge custoemr and product first year with main dataframe
combined_df = pd.merge(
    df, 
    customer_hist_df[['customer_id', 'sku', 'cust_first_year', 'prod_first_year']],
    how='left', on=['customer_id', 'sku'])