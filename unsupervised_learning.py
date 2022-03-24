import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA

# %matplotlib inline

# Multiple components analysis | PCA | Latent Direchlet Allocation | Kmodes

#%% Clustering Algorithms

class ClusteringModels():
    def __init__(self, model_name):
        self.valid_models = ['Kmeans', 'Kmodes', 'GMM']
        assert model_name in self.valid_models, "Please select one of: {} as a model".format(self.valid_models)
        
        self.model_name = model_name
        self.n_clusters = None
        self.model = []
        self.model_results = []
        
        model_descriptions = {
            'Kmeans':""" K-Means Summary Description:
    - Most popular clustering algorithm where each point fits in one cluster based on the linear distance to cluster centroids.
    - Only works on Ratio variables and the values must be standardized
    - The clusters are stochastic as they are dependent on the initial clusters""",
    
            'Kmodes':""" K-Modes Summary Description:
    - Similar in principle to Kmeans although uses modes and dissimilarity as the distance metric
    - Designed for use on categorical features and should be used before one-hot-encoding is applied
    - The clusters are stochastic as they are dependent on the initial clusters
    https://www.youtube.com/watch?v=b39_vipRkUo&ab_channel=AysanFernandes""",
    
            'GMM':""" Gaussian Mixture Model Summary Description:
    - Assumes the data comes from n independent Gaussian distributions
    - Only works on Ratio variables and the values must be standardized
    - Uses expectation maximization to find the most likely clusters
    - Assigns a probability to each point's cluster membership rather than hard clustering"""
            }
        self.model_description = model_descriptions[model_name]
    
    def fit(self, input_df, n_clusters):
        if self.model_name == 'Kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=34)
            model.fit(input_df)
            
        elif self.model_name == 'Kmodes':
            model = KModes(n_clusters=n_clusters, init='Huang')
            model.fit(input_df)
            
        elif self.model_name == 'GMM':
            model = GaussianMixture(n_components=n_clusters, random_state=34)
            model.fit(input_df)
        
        else:
            raise Exception("Please select one of: {} as a model".format(self.valid_models))
        
        self.n_clusters = n_clusters
        self.model = model
        
    def predict(self, input_df):
        if self.model_name == 'LDA':
            self.model.transform(input_df)
        else:
            return self.model.predict(input_df)

class DimensionalityReduction():
    def __init__(self, model_name):
        self.valid_models = ['PCA', 'MCA', 'LDA']
        assert model_name in self.valid_models, "Please select one of: {} as a model".format(self.valid_models)
        
        self.model_name = model_name


#%%
import data

si = data.SimpleImputer()

DF.X_test = si.fit_transform_df(DF.X_test, 'mode')

#%%


m = ClusteringModels('Kmodes')

b = DF.X_test.select_dtypes('O').copy()

print(m.model_description)

m.fit(b, 5)
a = m.predict(b)

#%%

# Apply k-modes to binary features using 'Huang' initialization to form clusters
def kmodes(X_train_input, X_test_input, n_clusters):
    binary_cols = [col for col in X_train_input.columns if X_train_input[col].isin([0, 1]).all()]
    
    km = KModes(n_clusters=n_clusters, init='Huang')
    km.fit(X_train_input[binary_cols])
    
    train_clusters = km.predict(X_train_input[binary_cols])
    test_clusters = km.predict(X_test_input[binary_cols])
    
    return train_clusters, test_clusters



lda = LatentDirichletAllocation(n_components=n_components, random_state=34,learning_method='batch')
            lda.fit(X_train_input) 
            X_train_output = pd.DataFrame(lda.transform(X_train_input),columns = 
                                          [output_name+'_LDA'+str(p+1) for p in range(n_components)],
                                          index=X_train_input.index)
            X_test_output = pd.DataFrame(lda.transform(X_test_input),columns = 
                                         [output_name+'_LDA'+str(p+1) for p in range(n_components)],
                                         index=X_test_input.index)
            return X_train_output, X_test_output

def get_kmeans():
    





def train_kmeans(input_X_train, input_X_test=None, n_clusters=5):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=34)
    kmeans_model.fit(input_X_train)

    pred_train_clusters = kmeans_model.predict(input_X_train)
    if input_X_test is not None:
        pred_test_clusters = kmeans_model.predict(input_X_test)
        return pred_train_clusters, pred_test_clusters, kmeans_model
    
    return pred_train_clusters, None, kmeans_model

def get_centroids(input_df, kmeans_model):
    closest, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, input_df)
    return input_df.iloc[closest]
    
def train_GMM(input_X_train, input_X_test=None, n_clusters=5):
    gm_model = GaussianMixture(n_components=n_clusters)
    gm_model.fit(input_X_train)
    
    pred_train_clusters = gm_model.predict(input_X_train)
    if input_X_test is not None:
        pred_test_clusters = gm_model.predict(input_X_test)
        return pred_train_clusters, pred_test_clusters, gm_model
        
    return pred_train_clusters, None, gm_model

def elbow_approach(input_X_train, cluster_algorithm='kmeans'):
    error = []
    
    n_clusters_to_consider = np.arange(1,11)
    
    for n_clusters in n_clusters_to_consider:
        if cluster_algorithm =='kmeans':
            _, _, kmeans_model = train_kmeans(input_X_train, None, n_clusters)
            error.append([n_clusters, kmeans_model.inertia_])
        elif cluster_algorithm =='GMM':
            _, _, gm_model = train_GMM(input_X_train, None, n_clusters)
            error.append([n_clusters, gm_model.bic(input_X_train)])
        else:
            print('Please select either kmeans or GMM')
            
    error = np.array(error)
    plt.xticks(n_clusters_to_consider)
    plt.plot(error[:,0], error[:,1])
    plt.show()


#%% PCA and visualization of clusters

def reduce_dimensionality(input_X_train, input_X_test, n_components=2):
    pca = PCA(n_components=n_components).fit(input_X_train)
    print('PCA Output:')
    print(pca.explained_variance_ratio_, 'Total:', 
          pca.explained_variance_ratio_.sum().round(2))
    X_train_PCA = pd.DataFrame(pca.transform(input_X_train),columns = 
                                  ['PC'+str(p+1) for p in range(n_components)],
                                  index=input_X_train.index)
    X_test_PCA = pd.DataFrame(pca.transform(input_X_test),columns = 
                                 ['PC'+str(p+1) for p in range(n_components)],
                                 index=input_X_test.index)
    return X_train_PCA, X_test_PCA


#%% Plot clusters

def plot_cluster_feat_means(input_df):
    cluster_feature_means = input_df.groupby(by='cluster').mean()
    
    # Show overview of bar plots together
    cdf_melt = cluster_feature_means.reset_index().melt(id_vars = 'cluster',
                          value_vars = list(cluster_feature_means.columns),
                          var_name = 'columns')
        
    sns.barplot(data=cdf_melt, x='cluster', y='value', hue='columns')
    plt.title('Average feature values by cluster')
    plt.legend(loc="lower center", bbox_to_anchor=(.5, -0.4), ncol=4, fontsize=10)
    
    return cluster_feature_means

def plot_ind_cluster_feature_means(input_df):
    cluster_feature_means = input_df.groupby(by='cluster').mean()
    
    for feature in cluster_feature_means.columns:
        plt.bar(cluster_feature_means[feature].index, cluster_feature_means[feature])
        plt.title('Average {} values by cluster'.format(feature))
        plt.xlabel("Cluster")
        plt.ylabel("Value")
        plt.show()
    return cluster_feature_means

#%% Load standardized data

import data
import exploratory_data_analysis as eda

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


#%% Find out correct number of clusters and fit models
# # Elbow approach to select n clusters
# elbow_approach(DF.X_train, 'kmeans')
# elbow_approach(DF.X_train, 'GMM')

# X_train_PCA, X_val_PCA = reduce_dimensionality(DF.X_train, DF.X_val)

# train_clusters, val_clusters, kmeans_model = train_kmeans(DF.X_train, DF.X_val, n_clusters=4)

# X_train_PCA['cluster'] = train_clusters
# X_val_PCA['cluster'] = val_clusters

# plt.scatter(X_train_PCA['PC1'], X_train_PCA['PC2'], c=train_clusters, cmap='Dark2')
# plt.show()

# #%% See relationship between clusters and features

# # Add the clusters to the data
# DF.X_train['cluster'] = train_clusters

# std_cluster_feature_means = plot_cluster_feat_means(DF.X_train)
# print()
# print('Standardized Average Feature Values: ')
# print(np.around(std_cluster_feature_means.reset_index(), 2).T)


# raw_cluster_feature_means = plot_ind_cluster_feature_means(DF.X_train)
# print()
# print('Average Feature Values: ')
# print(np.around(raw_cluster_feature_means.reset_index(), 2))


