import matplotlib
import matplotlib.pyplot as plt
from matplotlib import interactive

### Code to plot the evaluation of clustering methods
### Author: Jonathon Luiten
# 3 plot modes are {1:ami, 2:homogeneity, 3:completeness}
# If output_folder == None, then don't save
def plot_clustering_results(dict_of_eval_results, dataset_name, output_folder=None, display=False):

  matplotlib.rcParams.update({'font.size': 12, 'font.family': "Times New Roman"})
  matplotlib.rcParams['text.usetex'] = True
  lw = 4
  ms = 12.0
  interactive(True)

  styles = ['b-', 'b:', 'b--', 'r-', 'r:', 'r--', 'k-']
  y_axis_labels = ["AMI score", "Homogeneity score", "Completeness score"]

  for idx,(method, eval_res) in enumerate(dict_of_eval_results.iteritems()):
    for plot_mode in range(1, 4):
      plt.figure(plot_mode)
      if len(eval_res["graph"]) == 1:
        plt.plot([0, 50], [eval_res["graph"][:, plot_mode], eval_res["graph"][:, plot_mode]], styles[idx], linewidth=lw,
                 markersize=ms, label=method)
        dot_style = styles[idx][0] + 'x'
        plt.plot(eval_res["within"][plot_mode-1], dot_style, linewidth=lw, markersize=ms, label='_nolegend_')
      else:
        plt.plot(eval_res["graph"][:, 0], eval_res["graph"][:, plot_mode], styles[idx], linewidth=lw, markersize=ms,
                 label=method)
        dot_style = styles[idx][0] + 'o'
        plt.plot(eval_res["outlier_fraction"], eval_res["within"][plot_mode-1], dot_style, linewidth=lw, markersize=ms,
                 label='_nolegend_')

  for plot_mode in range(1, 4):
    plt.figure(plot_mode)
    ax = plt.gca()
    ax.grid(True)

    plt.xlabel("Outlier percentage")
    y_lab = y_axis_labels[plot_mode-1]
    dset_name = ' '.join(dataset_name.split('_'))
    plt.ylabel(y_lab)
    plt.title("Clustering " + y_lab + " results on " + dset_name)
    plt.legend()
    if output_folder:
      plt.savefig(output_folder + dataset_name + "_" + y_lab.split(' ')[0] + ".pdf")
    if display:
      plt.show()
    else:
      plt.close()

  if display:
    interactive(False)
    plt.show()
