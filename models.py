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
                n_estimators=self.hyperparameters['n_estimators'],
                objective='multiclass' if self.is_multiclass else 'binary',
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
                input_X_train, np.max(np.abs(model.coef_), axis=0).flatten())
        
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
            return np.full(shape=(len(input_X_test)), fill_value=self.model)
        
        if self.model_name == 'SVM':
            """SVMs do not directly provide probability estimates"""
            return self.model.predict(input_X_test)
        
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
    
#%% Model evaluation
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
    results['AUC'] = get_model_AUC(model_pred_probs, y_true)
    results['precision'] = get_model_precision(model_preds, y_true)
    results['recall'] = get_model_recall(model_preds, y_true)
    results['f-score'] = {'beta': beta,
                          'score':get_model_f_score(model_preds, y_true, beta=beta)}
    results['confusion_matrix'] = get_confusion_matrix(model_preds, y_true)
    
    return results

def get_roc_curve(model_pred_probs, y_true, plot_results=True):
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

