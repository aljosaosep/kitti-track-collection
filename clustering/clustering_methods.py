from hdbscan import HDBSCAN
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import mixture
from clustering_utils import membership_vector

### Code to cluster using HDBSCAN
### Author: Jonathon Luiten
# PCA is done first before clustering, the dimensionality becomes a parameter of the clustering
# Memory is used to speed up computations over multiple parameters, for the same n_components and min_samples,
# the computation intensive part of the clustering can be reused.
# Cluster labels for outliers are filled in with their closest label
# Glosh Outlier scores are calculated, for points given as outliers by the clustering +1 is added to the outlier score
# to be able to easily seperate outliers from inliers
def cluster_hdbscan(orig_ys, parameters, MemoryDir=None, classes=None):
  n_components, min_samples, min_cluster_size = parameters

  # Convert to np if not already
  orig_ys = np.array(orig_ys)
  orig_ys = orig_ys.astype('float64')

  # Load memory if needed
  if MemoryDir:
    savedMemory = Memory(MemoryDir + str(n_components) + "_" + str(min_samples) + "/")
  else:
    savedMemory = Memory(cachedir=None, verbose=0)

  # PCA to desired dimensionality
  pca = PCA(n_components=n_components)
  ys = pca.fit_transform(orig_ys)

  # Cluster using hdbscan
  clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=-2,
                      algorithm='boruvka_kdtree',
                      cluster_selection_method='eom', prediction_data=True, memory=savedMemory)
  cluster_labels = clusterer.fit_predict(ys)
  outlier_scores = clusterer.outlier_scores_

  # Increase outlier score of outlier points by 1
  ys_idx = np.arange(len(cluster_labels))
  outlier_idx = ys_idx[cluster_labels == -1]
  outlier_scores[outlier_idx] += 1

  # Assign cluster labels to outlier points
  soft_cluster_labels = membership_vector(clusterer, ys[outlier_idx])
  weak_cluster_labels = np.argmax(soft_cluster_labels, -1)
  cluster_labels[outlier_idx] = weak_cluster_labels

  return cluster_labels, outlier_scores


### Code to cluster using Kmeans
### Author: Jonathon Luiten
def cluster_kmeans(orig_ys, parameters, classes=None):
  n_components, num_clusters = parameters

  # Convert to np if not already
  orig_ys = np.array(orig_ys)
  orig_ys = orig_ys.astype('float64')

  # PCA to desired dimensionality
  pca = PCA(n_components=n_components)
  ys = pca.fit_transform(orig_ys)

  # Cluster using kmeans
  kmeans = KMeans(n_clusters=num_clusters, n_jobs=-1)
  cluster_labels = kmeans.fit_predict(ys)

  outlier_scores = None
  return cluster_labels, outlier_scores


### Code to cluster using GMM
### Author: Jonathon Luiten
def cluster_gmm(orig_ys, parameters, classes=None):
  n_components, num_clusters = parameters

  # Convert to np if not already
  orig_ys = np.array(orig_ys)
  orig_ys = orig_ys.astype('float64')

  # PCA to desired dimensionality
  pca = PCA(n_components=n_components)
  ys = pca.fit_transform(orig_ys)

  # Cluster using gmm
  gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type='full')
  gmm.fit(ys)
  cluster_labels = gmm.predict(ys)

  outlier_scores = None
  return cluster_labels, outlier_scores