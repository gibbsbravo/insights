import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import data
import exploratory_data_analysis as eda
import unsupervised_learning as ul
import models

# %matplotlib inline

#%% Load data & split into sets

input_df = data.load_file('data/train.csv')

# Remove empty variables
input_df.drop(
    ['PoolArea', 'PoolQC', '3SsnPorch', 'Alley', 'MiscFeature', 'LowQualFinSF', 'ScreenPorch', 'MiscVal'], axis=1, inplace=True)

# Set target_variable, train, validation, and test sets
target_name = 'SalePrice'
split_ratios = {'train' : 0.60,
                'validation' : 0.20,
                'test' : 0.20}
is_multiclass = False

DF = data.ModelData(input_df, target_name, split_ratios)

if is_multiclass:
    # Multiclass classification
    bin = data.BinEncoder()
    bin.fit(DF.y_train, 5)
    
    DF.y_train.update(bin.transform(DF.y_train))
    DF.y_val.update(bin.transform(DF.y_val))

else:
    # Binary classification
    DF.y_train.update(np.where(DF.y_train>200000, 1, 0))
    DF.y_val.update(np.where(DF.y_val>200000, 1, 0))


#%% Simple EDA

# Simple profiling
DF.X_train.info()

# Check for nulls
eda.prop_column_null(DF.X_train)[:20]
eda.prop_row_null(DF.X_train)[:20]

# Export summary HTML report of data profile
eda.create_html_data_profile(
    DF.X_train, 
    'outputs/data_profile.html', 
    overwrite=False)

#%% Data cleaning

# Check whether there are duplicates and remove
duplicate_rows = data.get_duplicate_rows(DF.X_train)

n_duplicated_rows = len(set(duplicate_rows.index).intersection(set(DF.X_train.index)))

if n_duplicated_rows > 0:
    DF.X_train.drop(duplicate_rows.index, axis=0, inplace=True)
    DF.X_train.reset_index(drop=True, inplace=True)
    print("Removed {} duplicated rows.".format(n_duplicated_rows))

# Fill null values
si = data.SimpleImputer()

DF.X_train = si.fit_transform_df(DF.X_train, strategy='mode')
DF.X_val = si.transform_df(DF.X_val)

# Categorical / mean encode variables
categorical_encodings_dict = {col : {'encoding' : 'mean', 'model' : None} 
                              for col in DF.X_train.select_dtypes('object').columns}

categorical_encodings_dict = data.fit_cat_encoding(DF.X_train, DF.y_train, categorical_encodings_dict)
DF.X_train = data.transform_cat_encoding(DF.X_train, categorical_encodings_dict)
DF.X_val = data.transform_cat_encoding(DF.X_val, categorical_encodings_dict)

# Remove Outliers
outliers = eda.get_isolation_forest_outliers(DF.X_train)
print('{} records ({:.2%}) identified as an outlier'.format(
    len(outliers['outlier_rows']),
    outliers['outlier_proportion']))

DF.X_train.drop(outliers['outlier_rows'].index,inplace=True)
DF.y_train.drop(outliers['outlier_rows'].index,inplace=True)
DF.X_train.reset_index(inplace=True, drop=True)
DF.y_train.reset_index(inplace=True, drop=True)

# Scale inputs
sc = data.StandardScaler()
    
DF.X_train = sc.fit_transform_df(DF.X_train)
DF.X_val = sc.transform_df(DF.X_val)

#%% Feature Engineering

# Feature importance
feat_importance_model = models.ClassificationModels('Random Forest',
                       hyperparameters={'max_depth':10})
feat_importance_model.fit(DF.X_train, DF.y_train)
feat_importance_model.plot_feature_importance()


#%% Clustering & dimensionality reduction

# Figure out how many clusters to create
ul.plot_elbow_approach(DF.X_train, 'Kmeans')
ul.plot_dendrogram(DF.X_train, 5)

# Train cluster model
kmeans_model = ul.ClusteringModels('Kmeans')
kmeans_model.fit(DF.X_train, n_clusters=3)

DF.X_train['Kmeans_cluster'] = kmeans_model.predict(DF.X_train)
DF.X_val['Kmeans_cluster'] = kmeans_model.predict(DF.X_val)

# Visualize the clusters
pca_model = ul.DimensionalityReduction('PCA')

pca_model.fit(DF.X_train.loc[:, DF.X_train.columns != 'Kmeans_cluster'], 2)
pca_dims = pca_model.transform(DF.X_train.loc[:, DF.X_train.columns != 'Kmeans_cluster'])

ul.plot_components_by_feature(pca_dims, DF.X_train['Kmeans_cluster'])
ul.plot_components_by_feature(pca_dims, DF.y_train)

# Add the clusters to the data as one-hot encoded
categorical_encodings_dict['Kmeans_cluster'] = {'encoding' : 'one-hot', 'model' : None} 
DF.X_train = data.transform_cat_encoding(
    DF.X_train,
    {key : value for key, value in categorical_encodings_dict.items() if key == 'Kmeans_cluster'})

DF.X_val = data.transform_cat_encoding(
    DF.X_val,
    {key : value for key, value in categorical_encodings_dict.items() if key == 'Kmeans_cluster'})


#%% Modelling 
# fit models 
model_name = 'LGBM'
gs_hyperparameters = {'num_leaves' : [4, 6, 8, 10, 20, 40]}

model = models.ClassificationModels(model_name)
model.gridsearch_hyperparameters(
    DF.X_train, DF.y_train, gs_hyperparameters)

model.fit(DF.X_train, DF.y_train)

# Get predictions
y_train_pred, y_train_pred_probs = model.predict(DF.X_train)
y_val_pred, y_val_pred_probs = model.predict(DF.X_val)

# evaluate models
train_performance = models.evaluate_model(y_train_pred, y_train_pred_probs, DF.y_train)
val_performance = models.evaluate_model(y_val_pred, y_val_pred_probs, DF.y_val)

models.plot_roc_curve(y_val_pred_probs, DF.y_val)

false_negative_records = models.get_false_negative_records(
    DF.X_val, y_val_pred, y_val_pred_probs, DF.y_val)
false_positive_records = models.get_false_positive_records(
    DF.X_val, y_val_pred, y_val_pred_probs, DF.y_val)

# Model Explainability


# model selection


# Model retraining on full set and export



#%% Save models




#%%

    
input_df = data.load_file('data/train.csv')

# Set target_variable, train, validation, and test sets
target_name = 'SalePrice'
split_ratios = {'train' : 0.60,
                'validation' : 0.20,
                'test' : 0.20}
is_multiclass = False

DF = data.ModelData(input_df, target_name, split_ratios)

if is_multiclass:
    # Multiclass classification
    bin = data.BinEncoder()
    bin.fit(DF.y_train, 5)
    
    DF.y_train.update(bin.transform(DF.y_train))
    DF.y_val.update(bin.transform(DF.y_val))

else:
    # Binary classification
    DF.y_train.update(np.where(DF.y_train>200000, 1, 0))
    DF.y_val.update(np.where(DF.y_val>200000, 1, 0))

removed_features = ['Id', 'PoolArea', 'PoolQC', '3SsnPorch', 'Alley',
                     'MiscFeature', 'LowQualFinSF', 'ScreenPorch', 'MiscVal']

pipeline_parameters = {
    'removed_features' : removed_features,
    'imputing_strategies' : {col : {'strategy' : 'mode', 'constant_value' : None, 'model' : None} 
                                  for col in DF.X_train.loc[:, DF.X_train.isna().any()] if col not in removed_features},
    'categorical_encodings' : {col : {'encoding' : 'mean', 'model' : None} 
                                  for col in DF.X_train.select_dtypes('object').columns if col not in removed_features},
    'include_engineered_feats' : True,
    'include_clustered_feats' : True,
    'cluster_models' : [
        {'cluster_name' : 'all_features_kmeans',
         'model_name' : 'Kmeans',
         'features' : [],
         'n_clusters' : 3,
         'model' : None},
        {'cluster_name' : 'neighborhood_kmeans',
         'model_name' : 'Kmeans',
         'features' : ['MSSubClass', 'Neighborhood', 'MSZoning'],
         'n_clusters' : 3,
         'model' : None}],
    
    'models' : {
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
    }


#%%

input_X_df = DF.X_train.copy()
input_y_df = DF.y_train.copy()
train=True


# def model_pipeline(input_X_df, input_y_df, pipeline_parameters, train=True):
pipeline_graph = []
model_results = []

pipeline_graph.append('Remove unused features')
input_X_df = data.drop_columns(input_X_df, pipeline_parameters['removed_features'])

pipeline_graph.append('Remove duplicates')
if train:
    # Check whether there are duplicates and remove
    duplicate_rows = data.get_duplicate_rows(input_X_df)

    n_duplicated_rows = len(set(duplicate_rows.index).intersection(set(DF.X_train.index)))

    if n_duplicated_rows > 0:
        DF.X_train.drop(duplicate_rows.index, axis=0, inplace=True)
        DF.X_train.reset_index(drop=True, inplace=True)
        print("Removed {} duplicated rows.".format(n_duplicated_rows))

pipeline_graph.append('Fill null values')
if train:
    data.fit_null_value_imputing(input_X_df, pipeline_parameters['imputing_strategies'])
input_X_df = data.transform_null_value_imputing(input_X_df, pipeline_parameters['imputing_strategies'])
    
pipeline_graph.append('Encode categorical features')
if train:
    categorical_encodings_dict = data.fit_cat_encoding(
        input_X_df, input_y_df, pipeline_parameters['categorical_encodings'])
input_X_df = data.transform_cat_encoding(input_X_df, categorical_encodings_dict)

pipeline_graph.append('Remove outliers')
outliers = eda.get_isolation_forest_outliers(input_X_df)
print('{} records ({:.2%}) identified as an outlier'.format(
    len(outliers['outlier_rows']),
    outliers['outlier_proportion']))

input_X_df.drop(outliers['outlier_rows'].index,inplace=True)
input_y_df.drop(outliers['outlier_rows'].index,inplace=True)
input_X_df.reset_index(inplace=True, drop=True)
input_y_df.reset_index(inplace=True, drop=True)

pipeline_graph.append('Scale Features')
if train:
    sc = data.StandardScaler()
    sc.fit_df(input_X_df)
input_X_df = sc.transform_df(input_X_df)

pipeline_graph.append('Train cluster models')
for cm_params in pipeline_parameters['cluster_models']:
    if train:
        cluster_model = ul.ClusteringModels(cm_params['model_name'])
        cluster_model.fit(
            input_X_df[cm_params['features']] if len(cm_params['features']) > 0 else input_X_df,
            n_clusters=cm_params['n_clusters'])
        cm_params['model'] = cluster_model

    input_X_df[cm_params['cluster_name']] = cluster_model.predict(
        input_X_df[cm_params['features']] if len(cm_params['features']) > 0 else input_X_df)

    pipeline_parameters['categorical_encodings'][cm_params['cluster_name']] = {
        'encoding' : 'one-hot', 'model' : None} 
    input_X_df = data.transform_cat_encoding(
        input_X_df,
        {key : value for key, value in categorical_encodings_dict.items() if key == cm_params['cluster_name']})

pipeline_graph.append('Train models')
for model_name, params in pipeline_parameters['models'].items():
    model = models.ClassificationModels(
        model_name, 
        hyperparameters=params['default_hyperparameters'])
    if params['gridsearch_hyperparameters'] is not None:
        model.gridsearch_hyperparameters(
            input_X_df, input_y_df, params['gridsearch_hyperparameters'])

    model.fit(input_X_df, input_y_df)
    pipeline_parameters['models'][model_name]['model'] = model

pipeline_graph.append('Get Model Predictions and Evaluate Performance')
for model_name, params in pipeline_parameters['models'].items():
    y_train_pred, y_train_pred_probs = params['model'].predict(input_X_df)
    performance = models.evaluate_model(y_train_pred, y_train_pred_probs, input_y_df)



