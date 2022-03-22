import pandas as pd
import numpy as np
import data

from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

#%%
input_df = pd.read_csv('data/train.csv')
target_name = 'SalePrice'
split_ratios = {'train' : 0.60,
                'validation' : 0.20,
                'test' : 0.20}

DF = data.ModelData(input_df, target_name, split_ratios)

#%% Profile data

def prop_column_null(input_df):
    """Returns the proportion of null values by column"""
    return (input_df.isnull().sum(axis = 0) / len(input_df)).sort_values(ascending=False)

def prop_row_null(input_df):
    """Returns the proportion of null values by row"""
    return (input_df.isnull().sum(axis = 1) / len(input_df.columns)).sort_values(ascending=False)


profile = ProfileReport(DF.X_train,
                        minimal=True)

profile.to_file("dataframe_profile.html")

profile = DF.X_train.profile_report(check_correlation_pearson=False,
    correlations={'pearson': False,
    'spearman': False,
    'kendall': False,
    'phi_k': False,
    'cramers': False,
    'recoded': False})



#%%


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





















input_df = pd.read_csv('data/train.csv')
target_name = 'SalePrice'
split_ratios = {'train' : 0.60,
                'validation' : 0.20,
                'test' : 0.20}

DF = data.ModelData(input_df, target_name, split_ratios)

sc = data.StandardScaler()
me = data.MeanEncoder()
bin = data.BinEncoder()

#%%

me.fit(DF.X_train.MSZoning, DF.y_train)
me.fit(DF.X_train.SaleType, DF.y_train)

me.model_parameters

a = mean_encode_column(DF.X_train.MSZoning, DF.y_train)

b = me.transform(DF.X_train.MSZoning)

bin.fit(DF.X_train.Id)

bin.transform(DF.X_train.Id)

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


balance_data(DF.X_train.select_dtypes(exclude=['object']), DF.y_train, 0.25, 'random_oversampling')


#%% Visualize categorical features



