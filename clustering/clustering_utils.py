import numpy as np
from hdbscan._prediction_utils import dist_membership_vector

# Method to load data files
def load_data(data_file=None, track_data_file=None):
  if data_file is not None:
    data = np.load(data_file + '.npz', allow_pickle=False, fix_imports=False)
  else:
    return None
  if track_data_file is not None:
    track_data = np.load(track_data_file + '.npz', allow_pickle=False, fix_imports=False)
  else:
    track_data = None
  return data, track_data

# Method to get the soft-assignments of points in the HDBSCAN clustering algorithm
def membership_vector(clusterer, points_to_predict):
  clusters = np.array(list(clusterer.condensed_tree_._select_clusters()
                           )).astype(np.intp)

  result = np.empty((points_to_predict.shape[0], clusters.shape[0]),
                    dtype=np.float64)

  for i in range(points_to_predict.shape[0]):
    distance_vec = dist_membership_vector(points_to_predict[i],
                                          clusterer.prediction_data_.exemplars,
                                          clusterer.prediction_data_.dist_metric)
    result[i] = distance_vec
    result[i] /= result[i].sum()
  return result
