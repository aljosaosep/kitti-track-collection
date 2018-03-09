import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score

### Code to evaluate clustering methods
### Author: Jonathon Luiten
# Assumes all points have been assigned to a valid cluster (not -1),
# but that points should be assigned as outliers (-1) with outlier scores greater than 1.
# if outlier_score == None, then no graph over outlier fraction is produced
def evaluate_clustering(orig_cluster_labels, orig_gt_labels, orig_outlier_scores=None):

  # Remove undefined tracks in the ground truth from the evaluation
  cluster_labels = orig_cluster_labels[orig_gt_labels!=-1]
  gt_labels = orig_gt_labels[orig_gt_labels!=-1]
  if orig_outlier_scores is not None:
    outlier_scores = orig_outlier_scores[orig_gt_labels!=-1]
    num_outliers = len(outlier_scores[outlier_scores > 1])
  else:
    outlier_scores = None
    num_outliers = 0

  # Check inputs
  assert len(cluster_labels)==len(gt_labels), "length gt and cluster labels must be the same"
  assert not any(gt_labels<0), "(-1) gt classes should be already removed from data "
  assert not any(cluster_labels < 0), "(-1) cluster labels should be assigned to the nearst class, but be in the 'num_outliers' highest 'outlier_scores'"

  # Sort labels by outlier scores
  if outlier_scores is not None:
    assert len(cluster_labels) == len(outlier_scores), "length outlier scores and cluster labels must be the same"
    sort_idx = np.argsort(outlier_scores,)
    cluster_labels = cluster_labels[sort_idx]
    gt_labels = gt_labels[sort_idx]
  else:
    assert num_outliers == 0, "with no outlier_scores, num_outliers must be 0"

  # Range of "within" evaluations for graphing
  graph = []
  for perc_out in range(0, 51, 1):
    curr_num = int((1-perc_out/100.0)*len(cluster_labels))
    curr_cluster_labels = cluster_labels[:curr_num]
    curr_gt_labels = gt_labels[:curr_num]
    ami = adjusted_mutual_info_score(curr_gt_labels, curr_cluster_labels)
    hom = homogeneity_score(curr_gt_labels, curr_cluster_labels)
    comp = completeness_score(curr_gt_labels, curr_cluster_labels)
    graph.append((perc_out,ami, hom, comp))
    if perc_out == 0 and outlier_scores is None:
      break
  graph = np.array(graph)

  # Basic properties of clustering
  num_clusters = len(np.unique(cluster_labels)) - 1
  num_labels = len(np.unique(gt_labels)) - 1
  outlier_fraction = float(num_outliers)/float(len(cluster_labels))*100
  num_inliers = len(cluster_labels) - num_outliers

  # Evaluation only on predicted labels ("within")
  curr_cluster_labels = cluster_labels[:num_inliers]
  curr_gt_labels = gt_labels[:num_inliers]
  ami = adjusted_mutual_info_score(curr_gt_labels, curr_cluster_labels)
  hom = homogeneity_score(curr_gt_labels, curr_cluster_labels)
  comp = completeness_score(curr_gt_labels, curr_cluster_labels)
  within = np.array((ami,hom,comp))

  # Evaluation on all labels ("outside")
  curr_cluster_labels = cluster_labels
  curr_cluster_labels[num_inliers:] = -1
  ami = adjusted_mutual_info_score(gt_labels, curr_cluster_labels)
  hom = homogeneity_score(gt_labels, curr_cluster_labels)
  comp = completeness_score(gt_labels, curr_cluster_labels)
  outside = np.array((ami, hom, comp))

  # Evaluation forcing all points to classes ("forced")
  forced = graph[0, 1:]

  # Return results
  eval_res_labels = ('num_clusters','num_labels', 'outlier_fraction','within','outside','forced','graph')
  eval_res_numbers = (num_clusters,   num_labels,  outlier_fraction,  within,  outside,  forced,  graph)
  eval_res = dict(zip(eval_res_labels,eval_res_numbers))
  return eval_res
