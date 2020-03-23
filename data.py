import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.model_selection import train_test_split

import seaborn as sns

import os

#%% Bucketize continuous variables

def equal_bin(input_feature, n_bins):
    sep = (input_feature.size/float(n_bins))*np.arange(1,n_bins+1)
    idx = sep.searchsorted(np.arange(input_feature.size))
    return idx[input_feature.argsort().argsort()]

def bin_feats(input_df, features_to_bin, n_bins=10):
    for feature in features_to_bin:
        input_df[feature] = equal_bin(input_df[feature], n_bins)

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

# Rename columns for wine 
#df.columns = ['target', "alcohol", "malic_acid", "ash" , "ash_alcalinity", "magnesium", "total_phenols",
# "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue",
# "OD280/OD315_diluted_wines", "proline"]

## Add null values to test
#df.proline.iloc[5:20] = np.nan
#df.hue.iloc[25:50] = np.nan