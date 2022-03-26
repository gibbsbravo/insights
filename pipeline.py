import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import data
import exploratory_data_analysis as eda
import unsupervised_learning
import models

# %matplotlib inline

#%%

input_df = pd.read_csv('data/train.csv')
input_df.drop(['PoolArea', 'PoolQC', '3SsnPorch', 'Alley', 'MiscFeature', 'LowQualFinSF', 'ScreenPorch', 'MiscVal'], axis=1, inplace=True)

target_name = 'SalePrice'
split_ratios = {'train' : 0.60,
                'validation' : 0.20,
                'test' : 0.20}

is_multiclass = False

DF = data.ModelData(input_df, target_name, split_ratios)


# Remove strings
DF.X_train = DF.X_train.select_dtypes(exclude=['object'])
DF.X_val = DF.X_val.select_dtypes(exclude=['object'])


if is_multiclass:
    # Multiclass cl=assification
    bin = data.BinEncoder()
    bin.fit(DF.y_train, 5)
    
    DF.y_train.update(bin.transform(DF.y_train))
    DF.y_val.update(bin.transform(DF.y_val))

else:
    # Binary classification
    DF.y_train.update(np.where(DF.y_train>200000, 1, 0))
    DF.y_val.update(np.where(DF.y_val>200000, 1, 0))

#Fill null values
si = data.SimpleImputer()

DF.X_train = si.fit_transform_df(DF.X_train, strategy='mean')
DF.X_val = si.fit_transform_df(DF.X_val, strategy='mean')

# Remove Outliers
outliers = eda.get_isolation_forest_outliers(DF.X_train)

DF.X_train.drop(outliers['outlier_rows'].index,inplace=True)
DF.y_train.drop(outliers['outlier_rows'].index,inplace=True)
DF.X_train.reset_index(inplace=True, drop=True)
DF.y_train.reset_index(inplace=True, drop=True)

# Scale inputs
sc = data.StandardScaler()

DF.X_train = sc.fit_transform_df(DF.X_train)
DF.X_val = sc.fit_transform_df(DF.X_val)



#%%
if False:
    classifier = models.ClassificationModels('LGBM', hyperparameters={'num_leaves' : 10, 'n_estimators': 100})
    classifier.gridsearch(DF.X_train, DF.y_train, {'num_leaves':[5, 15, 30, 60, 90]})
    
    classifier = models.ClassificationModels('Logistic Regression', hyperparameters={'C' : 1, 'penalty' : 'l2'})
    classifier.gridsearch(DF.X_train, DF.y_train, {'C':[0.01,0.1,1,10,100]})
    
    classifier = models.ClassificationModels('Random Forest', hyperparameters={'max_depth' : 1})
    classifier.gridsearch(DF.X_train, DF.y_train, {'max_depth':[1, 2, 8, 10, 20]})
    
    classifier = models.ClassificationModels('SVM', hyperparameters={'C' : 1})
    classifier.gridsearch(DF.X_train, DF.y_train, {'C':[0.01,0.1,10,], 'kernel':['linear', 'poly', 'rbf']})
    
    classifier.fit(DF.X_train, DF.y_train)
    
    model_preds = classifier.predict(DF.X_val)
    
    models.evaluate_model(model_preds, DF.y_val)
    











