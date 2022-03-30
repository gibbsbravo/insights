import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier

# %matplotlib inline

#%% Classification Models

class ClassificationModels():
    def __init__(self, model_name, hyperparameters={}, is_multiclass=False):
        self.valid_models = ['LGBM', 'Logistic Regression', 'Random Forest',
                             'SVM', 'KNN']
        assert model_name in self.valid_models, "Please select one of the following models: {}".format(self.valid_models)
        
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.is_multiclass = is_multiclass
        self.model = []
        self.feature_importance = pd.DataFrame([])
    
    def gridsearch_hyperparameters(self, input_X_train, input_y_train, gs_hyperparameters):
        if self.model_name == 'LGBM':
            model = lgb.LGBMClassifier(
                objective='multiclass' if self.is_multiclass else 'binary',
                random_state=34)
            clf = GridSearchCV(model, gs_hyperparameters).fit(
                input_X_train, input_y_train)
            
        elif self.model_name == 'Logistic Regression':
            model = LogisticRegression(random_state=34, max_iter=1000)
            clf = GridSearchCV(model, gs_hyperparameters).fit(
                    input_X_train, input_y_train)
        
        elif self.model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=34)
            clf = GridSearchCV(model, gs_hyperparameters).fit(
                    input_X_train, input_y_train)
        
        elif self.model_name == 'SVM':
            model = SVC(random_state=34)
            clf = GridSearchCV(model, gs_hyperparameters).fit(
                    input_X_train, input_y_train)
        
        elif self.model_name == 'KNN':
            model = KNeighborsClassifier()
            clf = GridSearchCV(model, gs_hyperparameters).fit(
                    input_X_train, input_y_train)
        
        print ("Best Parameters:", clf.best_params_)
        for param in gs_hyperparameters:
            self.hyperparameters[param] = clf.best_params_[param]
        
    def fit(self, input_X_train, input_y_train):
        if self.model_name == 'majority':
            model = input_y_train.mode().item()
        
        elif self.model_name == 'LGBM':
            model = lgb.LGBMClassifier(
                num_leaves=self.hyperparameters['num_leaves'], 
                n_estimators=200,
                objective='multiclass' if self.is_multiclass else 'binary',
                random_state=34)
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, model.feature_importances_)
        
        elif self.model_name == 'Logistic Regression':
            model = LogisticRegression(
                C=self.hyperparameters['C'],
                random_state=34, 
                max_iter=1000)
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, 
                np.abs(model.coef_) if (len(model.coef_.shape) == 1) else np.max(np.abs(model.coef_), axis=0).flatten())
        
        elif self.model_name == 'Random Forest':
            model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=self.hyperparameters['max_depth'],
                max_features='sqrt',
                n_jobs=4, 
                random_state=34)
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, model.feature_importances_)
            
        elif self.model_name == 'SVM':
            model = SVC(
                C=self.hyperparameters['C'],
                kernel=self.hyperparameters['kernel'],
                random_state=34)
            model.fit(input_X_train, input_y_train)
                    
        elif self.model_name == 'KNN':
            model = KNeighborsClassifier(
                n_neighbors=self.hyperparameters['n_neighbors'])
            model.fit(input_X_train, input_y_train)
            
        self.model = model
            
    def predict(self, input_X_test):
        if self.model_name == 'majority':
            return np.full(shape=(len(input_X_test)), fill_value=self.model), None
        
        elif self.model_name == 'SVM':
            """SVMs do not directly provide probability estimates"""
            return self.model.predict(input_X_test), None
        
        else:
            class_pred_probabilies = self.model.predict_proba(input_X_test)
            return np.argmax(class_pred_probabilies, axis=1), class_pred_probabilies
        
    def format_feature_importance_df(self, input_X_train, feature_importance):
        model_feature_importance = pd.DataFrame(feature_importance, 
                                  columns=["importance"],
                                  index = input_X_train.columns)
        model_feature_importance.sort_values(by='importance', ascending=False, inplace=True)
        return model_feature_importance
    
    def plot_feature_importance(self, max_features=10):
        if len(self.feature_importance) == 0:
            print("Need to fit model first")
        else:
            plt.bar(
                self.feature_importance.index[:max_features],
                self.feature_importance['importance'][:max_features])
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.xticks(rotation=45, ha='right')
            plt.show()
    
#%% Classification model evaluation
def get_model_accuracy(model_preds, y_true):
    return np.around((model_preds == y_true).sum() / len(y_true), 4)

def get_model_AUC(model_pred_probs, y_true):
    if len(set(y_true)) > 2:
        return np.around(
            roc_auc_score(
                y_true, 
                model_pred_probs,
                multi_class='ovr',
                average='weighted'
                )
            , 4)
    else:
        return np.around(roc_auc_score(y_true, model_pred_probs[:, 1]), 4)

def get_model_precision(model_preds, y_true):
    if len(set(y_true)) > 2:
        return np.around(
            precision_score(
                y_true, 
                model_preds,
                average='weighted'
                )
            , 4)
    else:
        return np.around(precision_score(y_true, model_preds), 4)
    

def get_model_recall(model_preds, y_true):
    if len(set(y_true)) > 2:
        return np.around(
            recall_score(
                y_true, 
                model_preds,
                average='weighted'
                )
            , 4)
    else:
        return np.around(recall_score(y_true, model_preds), 4)

def get_model_f_score(model_preds, y_true, beta=1):
    precision = get_model_precision(model_preds, y_true)
    recall = get_model_recall(model_preds, y_true)
    return np.around((1 + beta) * ((precision * recall) / ((beta * precision) + recall)), 4)

def get_confusion_matrix(model_preds, y_true):
    return pd.crosstab(
        y_true,
        np.around(model_preds),
        rownames=['True'],
        colnames=['Predicted'],
        margins=True)

def evaluate_model(model_preds, model_pred_probs, y_true, beta=1):
    results = {}
    results['accuracy'] = get_model_accuracy(model_preds, y_true)
    if model_pred_probs is not None:
        results['AUC'] = get_model_AUC(model_pred_probs, y_true)
    results['precision'] = get_model_precision(model_preds, y_true)
    results['recall'] = get_model_recall(model_preds, y_true)
    results['f-score'] = {'beta': beta,
                          'score':get_model_f_score(model_preds, y_true, beta=beta)}
    results['confusion_matrix'] = get_confusion_matrix(model_preds, y_true)
    
    return results

def plot_roc_curve(model_pred_probs, y_true, plot_results=True):
    assert len(set(y_true)) == 2, "ROC implementation only applies to binary classification"
    
    results = {}
    results['false_positive_rate'], results['true_positive_rate'], _ = roc_curve(
        y_true, model_pred_probs[:, 1])

    if plot_results:
        plt.figure(figsize=(8, 6))
        plt.title('Receiver Operating Characteristic Curve')
        plt.plot(
            results['false_positive_rate'], 
            results['true_positive_rate'],
            'b',
            label='AUC: {:.4f}'.format(
                get_model_AUC(model_pred_probs, y_true)))
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

def get_false_positive_records(input_X_df, model_preds, model_pred_probs, y_true):
    """Only works for binary classification"""
    return input_X_df.loc[(model_preds == 1) & (y_true == 0)]

def get_false_negative_records(input_X_df, model_preds, model_pred_probs, y_true):
    """Only works for binary classification"""
    return input_X_df.loc[(model_preds == 0) & (y_true == 1)]


#%%

import data
import exploratory_data_analysis as eda
import os

input_df = data.load_file('data/train.csv')

# Set target_variable, train, validation, and test sets
target_name = 'SalePrice'
split_ratios = {'train' : 0.60,
                'validation' : 0.20,
                'test' : 0.20}
DF = data.ModelData(input_df, target_name, split_ratios)


#%%

input_X_df = DF.X_train.copy()
input_y_df = DF.y_train.copy()

is_train_models=True
verbose=True
save_file=False


run_name = 'base_config_mean'
pipeline_parameters = data.load_file(
    os.path.join('data', 'pipeline_parameters', run_name + '.json'))

#%%

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

input_X_train = input_X_df.copy()
input_y_train = input_y_df.copy()

#%%

gs_hyperparameters = pipeline_parameters['models']['KNN']['gridsearch_hyperparameters']

gs_hyperparameters = {'alpha': [ 1.0]}

#%% Regression Models

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

class RegressionModels():
    def __init__(self, model_name, hyperparameters={}):
        self.valid_models = ['LGBM', 'Linear', 'Ridge', 'Random Forest', 'KNN']
        assert model_name in self.valid_models, "Please select one of the following models: {}".format(self.valid_models)
        
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self.model = []
        self.feature_importance = pd.DataFrame([])
    
    def gridsearch_hyperparameters(self, input_X_train, input_y_train, gs_hyperparameters):
        
        if self.model_name == 'Linear':
            # Linear has no hyperparameters to tune
            pass
        
        else:
            if self.model_name == 'LGBM':
                model = lgb.LGBMRegressor(random_state=34)
                reg = GridSearchCV(model, gs_hyperparameters).fit(
                    input_X_train, input_y_train)
            
            elif self.model_name == 'Ridge':
                model = Ridge(random_state=34, max_iter=1000)
                reg = GridSearchCV(model, gs_hyperparameters).fit(
                        input_X_train, input_y_train)
            
            elif self.model_name == 'Random Forest':
                model = RandomForestRegressor(random_state=34)
                reg = GridSearchCV(model, gs_hyperparameters).fit(
                        input_X_train, input_y_train)
            
            elif self.model_name == 'KNN':
                model = KNeighborsRegressor()
                reg = GridSearchCV(model, gs_hyperparameters).fit(
                        input_X_train, input_y_train)
            
            print ("Best Parameters:", reg.best_params_)
            for param in gs_hyperparameters:
                self.hyperparameters[param] = reg.best_params_[param]
            
    def fit(self, input_X_train, input_y_train):
        if self.model_name == 'mean':
            model = input_y_train.mean()
        
        elif self.model_name == 'LGBM':
            model = lgb.LGBMRegressor(
                num_leaves=self.hyperparameters['num_leaves'], 
                n_estimators=200,
                random_state=34)
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, model.feature_importances_)
        
        elif self.model_name == 'Linear':
            model = LinearRegression()
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, 
                np.abs(model.coef_) if (len(model.coef_.shape) == 1) else np.max(np.abs(model.coef_), axis=0).flatten())
            
        elif self.model_name == 'Ridge':
            model = Ridge(
                alpha=self.hyperparameters['alpha'],
                random_state=34,
                max_iter=1000)
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, 
                np.abs(model.coef_) if (len(model.coef_.shape) == 1) else np.max(np.abs(model.coef_), axis=0).flatten())
        
        elif self.model_name == 'Random Forest':
            model = RandomForestRegressor(
                n_estimators=200, 
                max_depth=self.hyperparameters['max_depth'],
                max_features='sqrt',
                n_jobs=4, 
                random_state=34)
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, model.feature_importances_)
            
        elif self.model_name == 'KNN':
            model = KNeighborsRegressor(
                n_neighbors=self.hyperparameters['n_neighbors'])
            model.fit(input_X_train, input_y_train)
            
        self.model = model
            
    def predict(self, input_X_test):
        if self.model_name == 'mean':
            return np.full(shape=(len(input_X_test)), fill_value=self.model)
        
        else:
            pred_values = self.model.predict(input_X_test)
            return pred_values
        
    def format_feature_importance_df(self, input_X_train, feature_importance):
        model_feature_importance = pd.DataFrame(feature_importance, 
                                  columns=["importance"],
                                  index = input_X_train.columns)
        model_feature_importance.sort_values(by='importance', ascending=False, inplace=True)
        return model_feature_importance
    
    def plot_feature_importance(self, max_features=10):
        if len(self.feature_importance) == 0:
            print("Need to fit model first")
        else:
            plt.bar(
                self.feature_importance.index[:max_features],
                self.feature_importance['importance'][:max_features])
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.xticks(rotation=45, ha='right')
            plt.show()