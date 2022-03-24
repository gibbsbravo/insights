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

def get_kmeans_centroids(input_df, kmeans_model):
    closest, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, input_df)
    return input_df.iloc[closest]

#%% Visualize optimal number of clusters

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

#%% Dimensionality reduction

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
        
        elif self.model_name == 'LDA':
            model = LatentDirichletAllocation(
                n_components=n_components, 
                learning_method='batch',
                random_state=34)
            model.fit(input_df)
            
        elif self.model_name == 'NMF':
            model = NMF(
                n_components=n_components,
                init='nndsvda',
                solver="mu", 
                max_iter=400,
                random_state=34)
            model.fit(input_df)
        
        else:
            raise Exception("Please select one of: {} as a model".format(self.valid_models))
            
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

#%% Plot clusters

def plot_components_by_feature(input_component_values, input_series):
    assert len(input_component_values.columns) == 2, "Please select two components"
    plt.scatter(
        input_component_values.iloc[:, 0],
        input_component_values.iloc[:, 1],
        c=input_series,
        cmap=sns.color_palette("viridis", as_cmap=True))
    plt.axis('off')
    plt.show()

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
