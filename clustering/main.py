import os
from clustering_utils import load_data
from clustering_methods import cluster_hdbscan, cluster_kmeans, cluster_gmm
from clustering_evaluation import evaluate_clustering
from clustering_plot_results import plot_clustering_results
from collections import OrderedDict
import time

def main():

  n_gt_classes = 33
  datasets = ['KITTI_raw']
  embedding_types = ['Triplet']
  parameters = [[(128, 14, 14), (128, n_gt_classes), (128, n_gt_classes)]]

  # second argument in values is if an enabled flag
  clustering_methods = OrderedDict([
                                    ('HDBSCAN', (cluster_hdbscan, True)),
                                    ('KMeans', (cluster_kmeans, True)),
                                    ('GMM', (cluster_gmm, True)),
                                    ])
  root_dir = '/home/' + os.environ['USER'] + '/vision/clustering/'
  dict_of_eval_results = OrderedDict()

  for dataset, curr_params in zip(datasets, parameters):
    for embedding in embedding_types:
      # Load data
      start = time.time()
      data_file = root_dir + 'data/' + dataset + '-' + embedding + '-OLD'
      track_data_file = root_dir + 'data/' + dataset + '-' + embedding + '-Tracks' + '-OLD'
      data, track_data = load_data(data_file, track_data_file)
      track_ys = track_data['track_ys']
      track_label_ids = track_data['track_label_ids']
      print "Loaded data. Elapsed:", time.time() - start

      for (method_name, (method_function, enabled)), method_params in zip(clustering_methods.iteritems(), curr_params):
        if not enabled:
          continue
        start = time.time()
        track_clusters, outlier_scores = method_function(track_ys, method_params)
        print method_name, 'clustering elapsed:', time.time() - start
        eval_res = evaluate_clustering(track_clusters, track_label_ids, outlier_scores)
        dict_of_eval_results[embedding + '-' + method_name] = eval_res
    plot_clustering_results(dict_of_eval_results, dataset, display=True)

if __name__ == "__main__":
  main()
