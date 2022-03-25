import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve

import data
import exploratory_data_analysis as eda

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#%% Load standardized data
input_df = pd.read_csv('data/train.csv')
input_df.drop(['PoolArea', 'PoolQC', '3SsnPorch', 'Alley', 'MiscFeature', 'LowQualFinSF', 'ScreenPorch', 'MiscVal'], axis=1, inplace=True)

target_name = 'SalePrice'
split_ratios = {'train' : 0.60,
                'validation' : 0.20,
                'test' : 0.20}


DF = data.ModelData(input_df, target_name, split_ratios)


# Remove strings
DF.X_train = DF.X_train.select_dtypes(exclude=['object'])
DF.X_val = DF.X_val.select_dtypes(exclude=['object'])

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

# Classification Models

#%% Naive Model

class ClassificationModels():
    def __init__(self, model_name, hyperparameters={}):
        self.model_name = model_name
        self.model = []
        self.feature_importance = pd.DataFrame([])
        self.hyperparameters = hyperparameters
    
    def format_feature_importance_df(self, input_X_train, feature_importance):
        model_feature_importance = pd.DataFrame(feature_importance, 
                                  columns=["importance"],
                                  index = input_X_train.columns)
        model_feature_importance.sort_values(by='importance', ascending=False, inplace=True)
        return model_feature_importance
    
    def gridsearch(self, input_X_train, input_y_train, gs_hyperparameters):
        if self.model_name == 'LGBM':
            model = lgb.LGBMClassifier(objective='binary', random_state=34)
            clf = GridSearchCV(model, gs_hyperparameters).fit(
                input_X_train, input_y_train)
            
        elif self.model_name == 'Logistic Regression':
            model = LogisticRegression(random_state=34)
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
        
        print ("Best Parameters:", clf.best_params_)
        for param in gs_hyperparameters:
            self.hyperparameters[param] = clf.best_params_[param]
            
        
    def fit(self, input_X_train, input_y_train):
        if self.model_name == 'majority':
            model = int(input_y_train.mean())
        
        elif self.model_name == 'LGBM':
            model = lgb.LGBMClassifier(
                num_leaves=self.hyperparameters['num_leaves'], 
                n_estimators=self.hyperparameters['n_estimators'],
                objective='binary',
                random_state=34)
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, model.feature_importances_)
        
        elif self.model_name == 'Logistic Regression':
            model = LogisticRegression(
                C=self.hyperparameters['C'],
                random_state=34)
            model.fit(input_X_train, input_y_train)
            
            self.feature_importance = self.format_feature_importance_df(
                input_X_train, np.abs(model.coef_).flatten())
        
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
            
        self.model = model
            
    def predict(self, input_X_test):
        if self.model_name == 'majority':
            return np.full(shape=(len(input_X_test)), fill_value=self.model)
        
        if self.model_name == 'SVM':
            """SVMs do not directly provide probability estimates"""
            return self.model.predict(input_X_test)
        
        else:
            return self.model.predict_proba(input_X_test)[:, 1]
    
#%% Model evaluation
def get_model_accuracy(model_preds, y_values):
    return np.around((np.around(model_preds) == y_values).sum() / len(y_values), 4)

def get_model_AUC(model_preds, y_values):
    return np.around(roc_auc_score(y_values, model_preds), 4)

def get_model_precision(model_preds, y_values):
    return np.around(precision_score(y_values, np.around(model_preds)), 4)

def get_model_recall(model_preds, y_values):
    return np.around(recall_score(y_values, np.around(model_preds)), 4)

def get_model_f_score(model_preds, y_values, beta=1):
    precision = get_model_precision(model_preds, y_values)
    recall = get_model_recall(model_preds, y_values)
    return np.around((1 + beta) * ((precision * recall) / ((beta * precision) + recall)), 4)

def get_confusion_matrix(model_preds, y_values):
    return pd.crosstab(
        y_values,
        np.around(model_preds),
        rownames=['True'],
        colnames=['Predicted'],
        margins=True)

def evaluate_model(model_preds, y_values, beta=1):
    results = {}
    results['accuracy'] = get_model_accuracy(model_preds, y_values)
    results['AUC'] = get_model_AUC(model_preds, y_values)
    results['precision'] = get_model_precision(model_preds, y_values)
    results['recall'] = get_model_recall(model_preds, y_values)
    results['f-score'] = {'beta': beta,
                          'score':get_model_f_score(model_preds, y_values, beta=beta)}
    results['confusion_matrix'] = get_confusion_matrix(model_preds, y_values)
    
    return results

# # Print Accuracy
# print('Model Results on Test Set:')
# print('    - Majority Classifier: {:.2f}%'.format(naive_test_accuracy*100))
# print('    - Random Forest: {:.2f}%'.format(rf_test_accuracy*100))
# print('    - Logistic Regression: {:.2f}%'.format(logr_test_accuracy*100))
# print('    - LightGBM : {:.2f}%'.format(lgbm_test_accuracy*100))

# # Plot Feature Importance 
# plt.bar(rf_feature_importance.index, rf_feature_importance['importance'])
# plt.title('Feature Importance')
# plt.xlabel('Features')
# plt.yticks([])
# plt.show()

# #%%  # Examine ROC curve for both models
# lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_pred_probs[:,1])
# logr_fpr, logr_tpr, _ = roc_curve(y_test, logr_pred_probs[:,1])
# rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_probs[:,1])

# plt.figure(figsize=(8, 6))
# plt.title('Receiver Operating Characteristic Curves')
# plt.plot(lgbm_fpr, lgbm_tpr, 'b', label = 'LGBM AUC = %0.4f' % lgbm_AUC)
# plt.plot(logr_fpr, logr_tpr, 'r', label = 'Logistic Regression AUC = %0.4f' % logr_AUC)
# plt.plot(rf_fpr, rf_tpr, 'g', label = 'Random Forest AUC = %0.4f' % rf_AUC)
# plt.legend(loc = 'lower right')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


#%%

classifier = ClassificationModels('LGBM', hyperparameters={'num_leaves' : 10, 'n_estimators': 100})
classifier.gridsearch(DF.X_train, DF.y_train, {'num_leaves':[5, 15, 30, 60, 90]})

classifier = ClassificationModels('Logistic Regression', hyperparameters={'C' : 1, 'penalty' : 'l2'})
classifier.gridsearch(DF.X_train, DF.y_train, {'C':[0.01,0.1,1,10,100]})

classifier = ClassificationModels('Random Forest', hyperparameters={'max_depth' : 1})
classifier.gridsearch(DF.X_train, DF.y_train, {'max_depth':[1, 2, 8, 10, 20]})

classifier = ClassificationModels('SVM', hyperparameters={'C' : 1})
classifier.gridsearch(DF.X_train, DF.y_train, {'C':[0.01,0.1,10,], 'kernel':['linear', 'poly', 'rbf']})

classifier.fit(DF.X_train, DF.y_train)

model_preds = classifier.predict(DF.X_val)

evaluate_model(model_preds, DF.y_val)
    
