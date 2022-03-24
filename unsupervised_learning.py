import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import dendrogram
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA, LatentDirichletAllocation, NMF

# %matplotlib inline

# Multiple components analysis | PCA | Latent Direchlet Allocation | Kmodes

#%% Clustering Algorithms

class ClusteringModels():
    def __init__(self, model_name):
        self.valid_models = ['Kmeans', 'Kmodes', 'Kprototypes', 'GMM', 'hierarchical', 'spectral']
        assert model_name in self.valid_models, "Please select one of: {} as a model".format(self.valid_models)
        
        self.model_name = model_name
        self.n_clusters = None
        self.model = []
        
        model_descriptions = {
            'Kmeans':""" K-Means Summary Description:
    - Most popular clustering algorithm where each point fits in one cluster based on the linear distance to cluster centroids.
    - Only works on ratio variables and the values must be standardized
    - The clusters are stochastic as they are dependent on the initial clusters""",
    
            'Kmodes':""" K-Modes Summary Description:
    - Similar in principle to Kmeans although uses modes and dissimilarity as the distance metric
    - Designed for use on categorical features and should be used before one-hot-encoding is applied
    - The clusters are stochastic as they are dependent on the initial clusters
    https://www.youtube.com/watch?v=b39_vipRkUo&ab_channel=AysanFernandes""",
    
            'Kprototypes':""" K-Prototypes Summary Description:
    - Combines Kmeans and Kmodes to handle mixed types
    - Requires ratio variables to be standardized and should be used before one-hot-encoding is applied""",
    
            'GMM':""" Gaussian Mixture Model Summary Description:
    - Assumes the data comes from n independent Gaussian distributions
    - Only works on Ratio variables and the values must be standardized
    - Uses expectation maximization to find the most likely clusters
    - Assigns a probability to each point's cluster membership rather than hard clustering""",
    
            'hierarchical':""" Hierarchical - Agglomerative Clustering (Ward's Method) Summary Description:
    - Recursively merges pair of clusters of sample data to minimize distance
    - Using Ward's distance has similar performance to K-Means in terms of detected clusters
    - Given it uses Ward's distance is still meant for ratio variables""",
    
            'spectral':""" Spectral Clustering Summary Description:
    - Computes a similarity graph, projects the data onto a low-dimensional space, create the clusters
    - Works well for non-convex clustering as there are no assumptions of the distribution of clusters
    - Is still meant for ratio variables after standardization"""}
        self.model_description = model_descriptions[model_name]
    
    def fit(self, input_df, n_clusters):
        if self.model_name == 'Kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=34)
            model.fit(input_df)
            
        elif self.model_name == 'Kmodes':
            model = KModes(n_clusters=n_clusters, init='Huang', random_state=34)
            model.fit(input_df)
        
        elif self.model_name == 'Kprototypes':
            categorical_cols_position = [input_df.columns.get_loc(col) for col in input_df.select_dtypes('O')]
            model = KPrototypes(n_clusters=n_clusters, init='Huang', random_state=34)
            model.fit(input_df, categorical=categorical_cols_position)
                        
        elif self.model_name == 'GMM':
            model = GaussianMixture(n_components=n_clusters, random_state=34)
            model.fit(input_df)
            
        elif self.model_name == 'hierarchical':
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
                affinity='euclidean')
            model.fit(input_df)
        
        elif self.model_name == 'spectral':
            model = SpectralClustering(
                n_clusters=n_clusters,
                assign_labels='discretize',
                random_state=34)
            model.fit(input_df)
        else:
            raise Exception("Please select one of: {} as a model".format(self.valid_models))
        
        self.n_clusters = n_clusters
        self.model = model
        
    def predict(self, input_df):
        if self.model_name == 'Kprototypes':
            categorical_cols_position = [input_df.columns.get_loc(col) for col in input_df.select_dtypes('O')]
            return self.model.predict(input_df, categorical=categorical_cols_position)
        
        elif self.model_name == 'hierarchical':
            assert len(input_df) == len(self.model.labels_), "Hierarchical clustering is non-parametric and only returns labels for values it was trained on"
            return self.model.labels_
        
        elif self.model_name == 'spectral':
            assert len(input_df) == len(self.model.labels_), "Spectral clustering is non-parametric and only returns labels for values it was trained on"
            return self.model.labels_
        
        else:
            return self.model.predict(input_df)

#%%

def elbow_approach(input_X_train, model, max_clusters=10):
    valid_models = ['Kmeans', 'Kmodes', 'Kprototypes', 'GMM']
    error = []
    
    for n_clusters in np.arange(1, max_clusters + 1):
        if model.model_name =='Kmeans':
            kmeans_model = ClusteringModels(model.model_name)
            kmeans_model.fit(input_X_train, n_clusters=n_clusters)
            error.append([n_clusters, kmeans_model.model.inertia_])
            
        elif model.model_name =='Kmodes':
            kmodes_model = ClusteringModels(model.model_name)
            kmodes_model.fit(input_X_train, n_clusters=n_clusters)
            error.append([n_clusters, kmodes_model.model.cost_])
        
        elif model.model_name =='Kprototypes':
            kprototypes_model = ClusteringModels(model.model_name)
            kprototypes_model.fit(
                input_X_train,
                n_clusters=n_clusters)
            error.append([n_clusters, kprototypes_model.model.cost_])  
            
        elif model.model_name =='GMM':
            gmm_model = ClusteringModels(model.model_name)
            gmm_model.fit(input_X_train, n_clusters=n_clusters)
            error.append([n_clusters, gmm_model.model.bic(input_X_train)])
        
        else:
            raise Exception("Please select one of: {} as a model".format(valid_models))
            
    error = np.array(error)
    plt.xticks(np.arange(1, max_clusters+1))
    plt.xlabel("Number of Clusters")
    plt.plot(error[:,0], error[:,1])
    plt.show()

def plot_dendrogram(input_df, depth=3):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(input_df)
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(linkage_matrix, truncate_mode="level", p=depth)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

#%%

class DimensionalityReduction():
    def __init__(self, model_name):
        self.valid_models = ['PCA', 'TSNE', 'LDA', 'NMF']
        assert model_name in self.valid_models, "Please select one of: {} as a model".format(self.valid_models)
        
        self.model_name = model_name
        self.n_components = None
        self.model = []
        self.model_information = {}

    def fit(self, input_df, n_components):
        if self.model_name == 'PCA':
            model = PCA(n_components=n_components, random_state=34)
            model.fit(input_df)
            self.model_information['explained_variance'] = np.around(model.explained_variance_ratio_, 2)
            self.model_information['total_explained_variance'] = model.explained_variance_ratio_.sum().round(2)
        
        elif self.model_name == 'TSNE':
            model = TSNE(
                n_components=2, 
                learning_rate='auto',
                init='random',
                random_state=34)
            model.fit(input_df)
            
        self.n_components = n_components
        self.model = model

    def transform(self, input_df):
        if self.model_name == 'TSNE':
            assert len(input_df) == len(self.model.embedding_), "T-SNE is non-parametric and only returns embeddings for values it was trained on"
            return pd.DataFrame(
                self.model.embedding_,
                columns = [self.model_name+str(p+1) for p in range(self.n_components)],
                index=input_df.index)
        else:
            return pd.DataFrame(
                self.model.transform(input_df),
                columns = [self.model_name+str(p+1) for p in range(self.n_components)],
                index=input_df.index)
        
        
#%%

pca = DimensionalityReduction('TSNE')

pca.fit(DF.X_train, 2)
a = pca.transform(DF.X_train)

pca.model_information


#%%


# si = data.SimpleImputer()
# DF.X_test = si.fit_transform_df(DF.X_test, 'mode')
# a = DF.X_test.select_dtypes('O').copy()

m = ClusteringModels('Kprototypes')

elbow_approach(
    pd.concat([DF.X_test.select_dtypes(exclude=['O']), DF.X_test[['MSZoning', 'SaleCondition']]], axis=1),
    m)

m.model.inertia_

#%%


elbow_approach(input_X_train, m, max_clusters=10)


#%%





#%%
m = ClusteringModels('Kmeans')
# b = DF.X_test.select_dtypes(exclude=['O']).copy()
b = DF.X_train

m.fit(b, 5)
a = m.predict(b, c.columns)

#%%



    

plot_dendrogram(DF.X_train, depth=4)
    


iris = load_iris()
X = iris.data




#%%

m = ClusteringModels('Kprototypes')

b = DF.X_test.select_dtypes(exclude=['O']).copy()
d = pd.concat([b, c], axis=1)

m.fit(d, 5, c.columns)
a = m.predict(d , c.columns)


#%%
import data

si = data.SimpleImputer()

DF.X_test = si.fit_transform_df(DF.X_test, 'mode')


#%%

# Apply k-modes to binary features using 'Huang' initialization to form clusters
def kmodes(X_train_input, X_test_input, n_clusters):
    binary_cols = [col for col in X_train_input.columns if X_train_input[col].isin([0, 1]).all()]
    
    km = KModes(n_clusters=n_clusters, init='Huang')
    km.fit(X_train_input[binary_cols])
    
    train_clusters = km.predict(X_train_input[binary_cols])
    test_clusters = km.predict(X_test_input[binary_cols])
    
    return train_clusters, test_clusters



# lda = LatentDirichletAllocation(n_components=n_components, random_state=34,learning_method='batch')
#             lda.fit(X_train_input) 
#             X_train_output = pd.DataFrame(lda.transform(X_train_input),columns = 
#                                           [output_name+'_LDA'+str(p+1) for p in range(n_components)],
#                                           index=X_train_input.index)
#             X_test_output = pd.DataFrame(lda.transform(X_test_input),columns = 
#                                          [output_name+'_LDA'+str(p+1) for p in range(n_components)],
#                                          index=X_test_input.index)
#             return X_train_output, X_test_output




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


