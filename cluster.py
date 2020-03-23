import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

train_data = pd.read_csv('data/std_train_data.csv')
test_data = pd.read_csv('data/std_test_data.csv')

X_train, y_train = train_data.loc[:, train_data.columns != 'target'], train_data['target']
X_test, y_test = test_data.loc[:, test_data.columns != 'target'], test_data['target']

#%% Clustering Algorithms

def train_kmeans(X_train_input, X_test_input=None, n_clusters=5):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=34)
    kmeans_model.fit(X_train_input)

    pred_train_clusters = kmeans_model.predict(X_train_input)
    if X_test_input is not None:
        pred_test_clusters = kmeans_model.predict(X_test_input)
        return pred_train_clusters, pred_test_clusters, kmeans_model
    
    return pred_train_clusters, None, kmeans_model
    
def train_GMM(X_train_input, X_test_input=None, n_clusters=5):
    gm_model = GaussianMixture(n_components=n_clusters)
    gm_model.fit(X_train_input)
    
    pred_train_clusters = gm_model.predict(X_train_input)
    if X_test_input is not None:
        pred_test_clusters = gm_model.predict(X_test_input)
        return pred_train_clusters, pred_test_clusters, gm_model
        
    return pred_train_clusters, None, gm_model

def elbow_approach(X_train_input, cluster_algorithm='kmeans'):
    error = []
    
    n_clusters_to_consider = np.arange(1,11)
    
    for n_clusters in n_clusters_to_consider:
        if cluster_algorithm =='kmeans':
            _, _, kmeans_model = train_kmeans(X_train_input, None, n_clusters)
            error.append([n_clusters, kmeans_model.inertia_])
        elif cluster_algorithm =='GMM':
            _, _, gm_model = train_GMM(X_train_input, None, n_clusters)
            error.append([n_clusters, gm_model.bic(X_train_input)])
        else:
            print('Please select either kmeans or GMM')
            
    error = np.array(error)
    plt.plot(error[:,0], error[:,1])
    plt.xticks(n_clusters_to_consider)

#%% PCA and visualization of clusters

def reduce_dimensionality(X_train_input, X_test_input, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(X_train_input)
    print('PCA Output:')
    print(pca.explained_variance_ratio_, 'Total:', 
          pca.explained_variance_ratio_.sum().round(2))
    X_train_PCA = pd.DataFrame(pca.transform(X_train_input),columns = 
                                  ['PC'+str(p+1) for p in range(n_components)],
                                  index=X_train_input.index)
    X_test_PCA = pd.DataFrame(pca.transform(X_test_input),columns = 
                                 ['PC'+str(p+1) for p in range(n_components)],
                                 index=X_test_input.index)
    return X_train_PCA, X_test_PCA

#%% 
# Elbow approach to select n clusters
elbow_approach(X_train, 'kmeans')

X_train_PCA, X_test_PCA = reduce_dimensionality(X_train, X_test)

train_clusters, test_clusters, model = train_kmeans(X_train, X_test, n_clusters=8)

X_train_PCA['cluster'] = train_clusters
X_test_PCA['cluster'] = test_clusters

plt.scatter(X_train_PCA['PC1'], X_train_PCA['PC2'], c=train_clusters, cmap='Dark2')

