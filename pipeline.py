import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

import os
import data
import exploratory_data_analysis as eda
import unsupervised_learning as ul
import models

# %matplotlib inline

#%% Define model pipeline

def model_pipeline(input_X_df,
                   input_y_df,
                   pipeline_parameters,
                   run_name='base_config', 
                   is_train_models=True,
                   verbose=True,
                   save_file=False):
    pipeline_graph = ["Pipeline Graph:"]
    if not is_train_models:
        run_name = pipeline_parameters['run_config']['run_name']
   
    if not is_train_models:
        input_X_df = data.align_feature_cols(
            input_X_df, 
            pipeline_parameters['run_config']['input_X_df_original_features'])
        
    input_X_df_original_features = input_X_df.columns
        
    pipeline_graph.append('Remove unused features')
    input_X_df = data.drop_columns(input_X_df, pipeline_parameters['removed_features'])
    
    pipeline_graph.append('Remove duplicates')
    if is_train_models:
        # Check whether there are duplicates and remove
        duplicate_rows = data.get_duplicate_rows(input_X_df)
    
        n_duplicated_rows = len(set(duplicate_rows.index).intersection(set(input_X_df.index)))
    
        if n_duplicated_rows > 0:
            input_X_df.drop(duplicate_rows.index, axis=0, inplace=True)
            input_X_df.reset_index(drop=True, inplace=True)
            print("Removed {} duplicated rows.".format(n_duplicated_rows))
    
    pipeline_graph.append('Fill null values')
    if is_train_models:
        data.fit_null_value_imputing(input_X_df, pipeline_parameters['imputing_strategies'])
    input_X_df = data.transform_null_value_imputing(
        input_X_df, 
        pipeline_parameters['imputing_strategies'],
        verbose=verbose)
    
    pipeline_graph.append('Encode categorical features')
    if is_train_models:
        pipeline_parameters['categorical_encodings'] = data.fit_cat_encoding(
            input_X_df, input_y_df, pipeline_parameters['categorical_encodings'])
    input_X_df = data.transform_cat_encoding(
        input_X_df, pipeline_parameters['categorical_encodings'])
    
    if is_train_models:
        pipeline_graph.append('Remove outliers')
        outliers = eda.get_isolation_forest_outliers(input_X_df)
        print('{} records ({:.2%}) identified as an outlier'.format(
            len(outliers['outlier_rows']),
            outliers['outlier_proportion']))
        
        input_X_df.drop(outliers['outlier_rows'].index,inplace=True)
        input_y_df.drop(outliers['outlier_rows'].index,inplace=True)
        input_X_df.reset_index(inplace=True, drop=True)
        input_y_df.reset_index(inplace=True, drop=True)
        
        input_X_df.drop(input_X_df.columns[input_X_df.isna().all()], axis=1, inplace=True)
        input_X_df.drop(input_X_df.columns[(input_X_df == 0).all()], axis=1, inplace=True)
    
    pipeline_graph.append('Standard Scale Features')
    if is_train_models:
        sc = data.StandardScaler()
        sc.fit_df(input_X_df)
        pipeline_parameters['feature_scaling_model'] = {'approach' : 'standard', 'model' : sc}
    input_X_df = pipeline_parameters['feature_scaling_model']['model'].transform_df(input_X_df)
    
    if not is_train_models:
        input_X_df = data.align_feature_cols(
            input_X_df,
            pipeline_parameters['run_config']['input_X_df_features_before_clustering'])
    
    input_X_df_features_before_clustering = input_X_df.columns
    
    for cm_params in pipeline_parameters['cluster_models']:
        cluster_features = []
        for feature in cm_params['features']:
            cluster_features.extend([col for col in input_X_df.columns if feature in col])
        
        if is_train_models:
            pipeline_graph.append('Train cluster model: {}'.format(cm_params['cluster_name']))
            cluster_model = ul.ClusteringModels(cm_params['model_name'])
            
            cluster_model.fit(
                input_X_df[cluster_features] if len(cluster_features) > 0 else input_X_df,
                n_clusters=cm_params['n_clusters'])
            cm_params['model'] = cluster_model
    
        pipeline_graph.append('Predict cluster model: {}'.format(cm_params['cluster_name']))
        input_X_df[cm_params['cluster_name']] = cm_params['model'].predict(
            input_X_df[cluster_features] if len(cluster_features) > 0 else input_X_df)
    
        pipeline_parameters['categorical_encodings'][cm_params['cluster_name']] = {
            'encoding' : 'one-hot', 'model' : None} 
        input_X_df = data.transform_cat_encoding(
            input_X_df,
            {key : value for key, value in pipeline_parameters['categorical_encodings'].items() 
             if key == cm_params['cluster_name']})

    if not is_train_models:
        input_X_df = data.align_feature_cols(
            input_X_df,
            pipeline_parameters['run_config']['input_X_df_transformed_features'])

    input_X_df_transformed_features = input_X_df.columns

    if is_train_models:
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
        params['performance'] = models.evaluate_model(y_train_pred, y_train_pred_probs, input_y_df)
    
    pipeline_parameters['run_config'] = {
        'run_name' : run_name,
        'date' : datetime.datetime.now().strftime("%d-%b-%Y - %I:%M %p"),
        'is_train_models' : is_train_models,
        'input_X_df_transformed_shape' : input_X_df.shape,
        'input_y_df_transformed_shape' : input_y_df.shape,
        'verbose': verbose,
        'input_X_df_original_features' : input_X_df_original_features,
        'input_X_df_features_before_clustering' : input_X_df_features_before_clustering,
        'input_X_df_transformed_features' : input_X_df_transformed_features,
        'pipeline_graph' : pipeline_graph}
    
    
    if save_file:
        data.save_file(
            pipeline_parameters, 
            'outputs/{}_{}_{}.pickle'.format(
                datetime.datetime.now().strftime("%Y.%m.%d.%I_%M_%p"),
                run_name,
                'train' if is_train_models else 'pred'),
            overwrite=False)
    
    return pipeline_parameters
        
# %%
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

data.create_parameters_template(
    DF.X_train, 
    output_file_path='data/pipeline_parameters/pipeline_parameters_template.json', 
    overwrite=False)

#%%

is_train_models = False

if is_train_models:
    input_X_df = DF.X_train.copy()
    input_y_df = DF.y_train.copy()
    run_name = 'base_config_mean'
    pipeline_parameters = data.load_file(
        os.path.join('data', 'pipeline_parameters', run_name + '.json'))
    
else:
    input_X_df = DF.X_val.copy()
    input_y_df = DF.y_val.copy()
    saved_parameters_file_name = '2022.03.29.05_24_PM_base_config_mean_train.pickle'
    pipeline_parameters = data.load_file(os.path.join('outputs', saved_parameters_file_name))


pipeline_parameters = model_pipeline(input_X_df,
               input_y_df,
               pipeline_parameters=pipeline_parameters,
               run_name=run_name, 
               is_train_models=is_train_models,
               verbose=True,
               save_file=True)

#%%


file_names = ['outputs/2022.03.29.05_24_PM_base_config_mean_pred.pickle', 
              'outputs/2022.03.29.05_22_PM_base_config_one_hot_pred.pickle']

results = []

for file_name in file_names:
    pipeline_parameters = data.load_file(file_name)
    
    for model_name, values in pipeline_parameters['models'].items():
        model_performance = values['performance']
        model_performance['model_name'] = model_name
        model_performance['file_name'] = file_name
        model_performance['config_name'] = pipeline_parameters['run_config']['run_name']
        model_performance['X_input_shape'] = pipeline_parameters['run_config']['input_X_df_shape']
        model_performance['is_train_models'] = pipeline_parameters['run_config']['is_train_models']
        
        model_performance['f{}-score'.format(model_performance['f-score']['beta'])] = model_performance['f-score']['score']
        model_performance = {key : value for key, value in model_performance.items() if key not in ['confusion_matrix', 'f-score']}
        
        results.append(model_performance)

results_df = pd.DataFrame(results)


#%%

input_train_cols = pipeline_parameters['run_config']['input_X_df_original_features']


# print("\n--------------------------\n".join(pipeline_parameters['run_config']['pipeline_graph']))


#%% EDA and feature engineering

##%% Simple EDA

# # Simple profiling
# DF.X_train.info()

# # Check for nulls
# eda.prop_column_null(DF.X_train)[:20]
# eda.prop_row_null(DF.X_train)[:20]

# # Export summary HTML report of data profile
# eda.create_html_data_profile(
#     DF.X_train, 
#     'outputs/data_profile.html', 
#     overwrite=False)

# #%% Feature Engineering

# # Feature importance
# feat_importance_model = models.ClassificationModels('Random Forest',
#                        hyperparameters={'max_depth':10})
# feat_importance_model.fit(DF.X_train, DF.y_train)
# feat_importance_model.plot_feature_importance()


# #%%
# # Visualize the clusters
# pca_model = ul.DimensionalityReduction('PCA')

# pca_model.fit(DF.X_train.loc[:, DF.X_train.columns != 'Kmeans_cluster'], 2)
# pca_dims = pca_model.transform(DF.X_train.loc[:, DF.X_train.columns != 'Kmeans_cluster'])

# ul.plot_components_by_feature(pca_dims, DF.X_train['Kmeans_cluster'])
# ul.plot_components_by_feature(pca_dims, DF.y_train)

