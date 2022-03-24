import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve

import data
import exploratory_data_analysis as eda
import unsupervised_learning

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

DF.y_train = np.where(DF.y_train>200000, 1, 0)
DF.y_val = np.where(DF.y_val>200000, 1, 0)

#Fill null values
si = data.SimpleImputer()

DF.X_train = si.fit_transform_df(DF.X_train, strategy='mean')
DF.X_val = si.fit_transform_df(DF.X_val, strategy='mean')

# Scale inputs
sc = data.StandardScaler()

DF.X_train = sc.fit_transform_df(DF.X_train)
DF.X_val = sc.fit_transform_df(DF.X_val)

# Remove Outliers
outliers = eda.get_isolation_forest_outliers(DF.X_train)

DF.X_train.drop(outliers['outlier_rows'].index,inplace=True)
DF.X_train.reset_index(inplace=True, drop=True)

#%% Classification Models

#%% Naive Model
def majority_class(y_train_input, y_test_input):
    majority_class = int(y_train_input.mean()) 
    naive_train_preds = [majority_class] * len(y_train_input)
    naive_test_preds = [majority_class] * len(y_test_input)
    
    train_accuracy = (naive_train_preds == y_train_input).sum() / len(y_train_input)
    test_accuracy = (naive_test_preds == y_test_input).sum() / len(y_test_input)
    
    return naive_train_preds, naive_test_preds, train_accuracy, test_accuracy, majority_class

#%% Train LightGBM model with optional gridsearch hyperparameter tuning
# Returns key evaluation metrics
def light_gbm(X_train_input, y_train_input, X_test_input, y_test_input,
              num_leaves=15, n_estimators=250, gridsearch=False):

    if gridsearch:
        # Gridsearch optimal number of leaves
        parameters = {'num_leaves':[5,15,30,60,90]}

        lgbm = lgb.LGBMClassifier(num_leaves=num_leaves, n_estimators=n_estimators,
                                  objective='binary', random_state=34)
        clf = GridSearchCV(lgbm, parameters).fit(X_train_input, y_train_input)
        print ("Best Parameters:", clf.best_params_)

        num_leaves = clf.best_params_['num_leaves']
        
    lgbm_clf = lgb.LGBMClassifier(
            num_leaves=num_leaves, n_estimators=n_estimators, objective='binary',
            random_state=34).fit(X_train_input,y_train_input.values.ravel())
    
    lgbm_train_preds = lgbm_clf.predict(X_train_input)
    lgbm_test_preds = lgbm_clf.predict(X_test_input)
    
    lgbm_pred_probs = lgbm_clf.predict_proba(X_test_input)
    y_pred = lgbm_pred_probs[:,1]
    
    AUC = roc_auc_score(y_test_input, y_pred)
    train_accuracy = lgbm_clf.score(X_train_input, y_train_input) 
    test_accuracy = lgbm_clf.score(X_test_input, y_test_input)
    
    lgbm_feature_importance = lgbm_clf.feature_importances_
    
    # Check the variable importance
    lgbm_feature_importance = lgbm_clf.feature_importances_
    lgbm_feature_importance = pd.DataFrame(lgbm_feature_importance, 
                              columns=["importance"],
                              index = X_train_input.columns)
    lgbm_feature_importance.sort_values(by='importance',ascending=False,inplace=True)
    
    return (lgbm_train_preds, lgbm_test_preds, AUC, train_accuracy,
            test_accuracy, lgbm_pred_probs, lgbm_clf, lgbm_feature_importance)


#%% Train logistic regression model with optional gridsearch hyperparameter tuning

def logistic_regression(X_train_input, y_train_input, X_test_input, y_test_input,
                        C=1, gridsearch=False):
    if gridsearch:
        # Gridsearch C value
        parameters = {'C':[0.01,0.1,1,10,100]}

        clf = GridSearchCV(LogisticRegression(C=C), parameters).fit(
                X_train_input, y_train_input)
        print ("Best Parameters:", clf.best_params_)

        C = clf.best_params_['C']

    logr_clf = LogisticRegression(C=C).fit(X_train_input, y_train_input)
    
    logr_train_preds = logr_clf.predict(X_train_input)
    logr_test_preds = logr_clf.predict(X_test_input)

    logr_pred_probs = logr_clf.predict_proba(X_test_input)
    y_pred = logr_pred_probs[:,1]
    
    AUC = roc_auc_score(y_test_input, y_pred)
    train_accuracy = logr_clf.score(X_train_input, y_train_input) 
    test_accuracy = logr_clf.score(X_test_input, y_test_input)
    
    return (logr_train_preds, logr_test_preds, AUC, train_accuracy,
            test_accuracy, logr_pred_probs, logr_clf)

#%% Train random forest model with optional gridsearch hyperparameter tuning

def random_forest(X_train_input, y_train_input, X_test_input, y_test_input,
                  max_depth=10, gridsearch=False):
    
    if gridsearch:
        # Gridsearch max tree depth
        n_features = len(X_train_input.columns)
        parameters = {'max_depth':[int(n_features*0.25), int(n_features*0.50), 
                                   int(n_features*0.75), int(n_features), None]}

        clf = GridSearchCV(RandomForestClassifier(n_estimators=200, max_depth=max_depth,
            max_features='sqrt', n_jobs=4, random_state=34), parameters).fit(
                X_train_input, y_train_input)
        print ("Best Parameters:", clf.best_params_)

        max_depth = clf.best_params_['max_depth']
    
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=max_depth,
            max_features='sqrt', n_jobs=4, random_state=34)
        
    rf_clf.fit(X_train, y_train)
    
    rf_train_preds = rf_clf.predict(X_train_input)
    rf_test_preds = rf_clf.predict(X_test_input)

    rf_pred_probs = rf_clf.predict_proba(X_test_input)
    y_pred = rf_pred_probs[:,1]
    
    AUC = roc_auc_score(y_test_input, y_pred)
    train_accuracy = rf_clf.score(X_train_input, y_train_input) 
    test_accuracy = rf_clf.score(X_test_input, y_test_input)
    
    # Check the feature importance
    rf_feature_importance = rf_clf.feature_importances_
    rf_feature_importance = pd.DataFrame(rf_feature_importance, 
                              columns=["importance"],
                              index = X_train_input.columns)
    rf_feature_importance.sort_values(by='importance', ascending=False, inplace=True)

    return (rf_train_preds, rf_test_preds, AUC, train_accuracy,
            test_accuracy, rf_pred_probs, rf_clf, rf_feature_importance)




#%%

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PyTorchDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.values).float()
        self.y = torch.from_numpy(y_train).type(torch.LongTensor) #.view(-1, 1)
        self.n_samples = len(X_train)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

train_dataloader = DataLoader(
    PyTorchDataset(DF.X_train, DF.y_train),
    batch_size=32,
    shuffle=True)

val_dataloader = DataLoader(
    PyTorchDataset(DF.X_val, DF.y_val),
    batch_size=32,
    shuffle=True)

# dataiter = iter(train_dataloader)
# data = dataiter.next()

#%%
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(p=0.20),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes))
        
    def forward(self, x):
        x = self.network(x)
        return x

input_size = DF.X_train.shape[1]
hidden_size = 32
num_classes = 2
num_epochs = 100

batch_size = 32
learning_rate = 0.001

print_every_n_iters = 10

model = MLP(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(num_epochs):
    for idx, (features, targets) in enumerate(train_dataloader):
        # Transfer to GPU if enabled
        if device.type == 'gpu':
            features, targets = features.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (idx + 1) % print_every_n_iters == 0:
            print("Epoch: {} | Step {}: Loss {:.4f}".format(
                epoch+1, idx+1, loss.item()))

#%%

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for idx, (features, targets) in enumerate(val_dataloader):
        # Transfer to GPU if enabled
        if device.type == 'gpu':
            features, targets = features.to(device), targets.to(device)
            
        outputs = model(features)
        
        prob, predictions = torch.max(outputs, axis=1)
        n_samples += targets.shape[0]
        n_correct += (predictions == targets).sum().item()

    accuracy = n_correct / n_samples
    print('Accuracy: {:.2%}'.format(accuracy))
    

#%%

DF.y_val.sum() / len(DF.y_val)


#%%

model2 = MLP(input_size, hidden_size, num_classes)

model2.load_state_dict(model.state_dict())

#%%




#%%

model2.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for idx, (features, targets) in enumerate(val_dataloader):
        # Transfer to GPU if enabled
        if device.type == 'gpu':
            features, targets = features.to(device), targets.to(device)
            
        outputs = model2(features)
        
        prob, predictions = torch.max(outputs, axis=1)
        n_samples += targets.shape[0]
        n_correct += (predictions == targets).sum().item()

    accuracy = n_correct / n_samples
    print('Accuracy: {:.2%}'.format(accuracy))
    
#%% Load standardized data

train_data = pd.read_csv('data/std_train_data.csv')
test_data = pd.read_csv('data/std_test_data.csv')

X_train, y_train = train_data.loc[:, train_data.columns != 'target'], train_data['target']
X_test, y_test = test_data.loc[:, test_data.columns != 'target'], test_data['target']



#%% Train models
    
# Majority class naive classifier
(naive_train_preds, naive_test_preds, naive_train_accuracy,
 naive_test_accuracy, naive_majority_class) = majority_class(y_train, y_test)

# LGBM model
(lgbm_train_preds, lgbm_test_preds, lgbm_AUC, lgbm_train_accuracy,
lgbm_test_accuracy, lgbm_pred_probs, lgbm_clf, lgbm_feature_importance) = light_gbm(
        X_train, y_train, X_test, y_test, num_leaves=15, n_estimators=250, gridsearch=False)

# Random Forests
(rf_train_preds, rf_test_preds, rf_AUC, rf_train_accuracy,
rf_test_accuracy, rf_pred_probs, rf_clf, rf_feature_importance) = random_forest(
                    X_train, y_train, X_test, y_test, max_depth=10, gridsearch=False)

# Logistic Regression Model
(logr_train_preds, logr_test_preds, logr_AUC, logr_train_accuracy,
logr_test_accuracy, logr_pred_probs, logr_model) = logistic_regression(
        X_train, y_train, X_test, y_test, C=1, gridsearch=False)

#%% Evaluate models

# Print Accuracy
print('Model Results on Test Set:')
print('    - Majority Classifier: {:.2f}%'.format(naive_test_accuracy*100))
print('    - Random Forest: {:.2f}%'.format(rf_test_accuracy*100))
print('    - Logistic Regression: {:.2f}%'.format(logr_test_accuracy*100))
print('    - LightGBM : {:.2f}%'.format(lgbm_test_accuracy*100))

# Plot Feature Importance 
plt.bar(rf_feature_importance.index, rf_feature_importance['importance'])
plt.title('Feature Importance')
plt.xlabel('Features')
plt.yticks([])
plt.show()

#%% Advanced Evaluation
# Examine confusion matrices for both models and performance metrics
model_names = ['Light GBM','Logistic Regression', 'Random Forest']
model_predictions = [lgbm_test_preds, logr_test_preds, rf_test_preds]
model_AUC = [lgbm_AUC, logr_AUC, rf_AUC]

f_beta = 1

print('Accuracy of Majority classifier on test set: {:.4f}'
     .format(naive_test_accuracy))
print()

for model_name, model_predictions, model_AUC in zip(model_names, model_predictions, model_AUC):
    print(model_name,'Model Evaluation:')
    print('Accuracy: {:.4f}'.format(accuracy_score(y_test, model_predictions)))
    print('AUC: {:.4f}'.format(model_AUC))
    
    precision = precision_score(y_test, model_predictions)
    print('Precision: {:.4f}'.format(precision))
    
    recall = recall_score(y_test, model_predictions)
    print('Recall: {:.4f}'.format(recall))
    print('F-score (Beta: {:.0f}): {:.4f}'.format(f_beta, (1+f_beta)*((precision*recall)/((f_beta*precision)+recall))))
    print()

    print('Confusion Matrix:')
    print(pd.crosstab(y_test, model_predictions, rownames=['True'], colnames=['Predicted'], margins=True))
    print()


#%%  # Examine ROC curve for both models
lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_pred_probs[:,1])
logr_fpr, logr_tpr, _ = roc_curve(y_test, logr_pred_probs[:,1])
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_probs[:,1])


plt.figure(figsize=(8, 6))
plt.title('Receiver Operating Characteristic Curves')
plt.plot(lgbm_fpr, lgbm_tpr, 'b', label = 'LGBM AUC = %0.4f' % lgbm_AUC)
plt.plot(logr_fpr, logr_tpr, 'r', label = 'Logistic Regression AUC = %0.4f' % logr_AUC)
plt.plot(rf_fpr, rf_tpr, 'g', label = 'Random Forest AUC = %0.4f' % rf_AUC)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(MLP, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.relu1 = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size, hidden_size)
#         self.relu2 = nn.ReLU()
#         self.l3 = nn.Linear(hidden_size, num_classes)
        
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu1(out)
#         out = self.l2(out)
#         out = self.relu2(out)
#         out = self.l3(out)
#         return out


