import numpy as np
from scipy import stats
import seaborn as sn
import matplotlib.pyplot as plt
import dataset,utils

def pvalue_matrix(in_path,clf_type="RF",metric="acc"):
    result_dict=get_result_dict(in_path)
    metric_dict=compute_metric(result_dict,metric)
    all_clfs=list(metric_dict.values())[0].keys()
    other_clfs=[clf_i for clf_i in all_clfs
                    if(clf_i!=clf_type) ]
    data=list(metric_dict.keys())
    sig_matrix=[]
    for data_i in data:
        metric_i=metric_dict[data_i][clf_type]
        sig_i=[]
        for clf_j in other_clfs:
            metric_j=metric_dict[data_i][clf_j]
            diff_i=np.mean(metric_i)-np.mean(metric_j)
#            print(data_i,clf_j,diff_i)
            pvalue=stats.ttest_ind(metric_i,metric_j,
                               equal_var=False)[1]
            sign_ij= int(pvalue<0.05) * np.sign(diff_i)
            sig_i.append(sign_ij)
        sig_matrix.append(sig_i)
    sig_matrix=np.array(sig_matrix)
    print(sig_matrix.shape)
    heatmap(matrix=sig_matrix,
            x_labels=other_clfs,
            y_labels=data)
    print(sig_matrix)

def get_result_dict(in_path):
    @utils.DirFun(out_arg=None)
    def helper(in_path):
        output={}
        for path_i in utils.top_files(in_path):
            name_i=path_i.split("/")[-1]
            if(name_i!="splits"):
                result_path_i=f"{path_i}/results"
                result_i=dataset.read_result_group(result_path_i)
                output[name_i]=result_i
        return output
    return helper(in_path)

def compute_metric(result_dict,metric):
    return { data_i:{name_j:result_j.get_metric(metric) 
                       for name_j,result_j in dict_i.items()}
              for data_i,dict_i in result_dict.items()}

def heatmap(matrix,
            x_labels,
            y_labels,
            title="Statistical significance (RF)"):
    fig, ax = plt.subplots()
    ax=sn.heatmap(matrix,
                  cmap="RdBu_r",
                  linewidth=0.5,
                  cbar=False,
                  ax=ax)
    y_labels.sort()
    ax.set_xticklabels(x_labels,rotation = 90)
    ax.set_yticklabels(y_labels,rotation = 0)
    ax.set_title(title,fontsize=10)
    plt.tight_layout()
    plt.show()
#    return ax.get_figure()

in_path="uci_exp/exp"
pvalue_matrix(in_path,metric="acc")