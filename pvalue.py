import numpy as np
from scipy import stats
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
import dataset,eval,utils
utils.silence_warnings()

def pvalue_matrix(in_path,clf_type="RF",metric="acc"):
    result_dict=eval.get_result_dict(in_path)
    metric_dict=result_dict.compute_metric(metric)
    all_clfs=result_dict.clfs()
    other_clfs=[clf_i for clf_i in all_clfs
                    if(clf_i!=clf_type) ]
#    data=list(metric_dict.keys())
    sig_matrix=[]
    for data_i in result_dict.data():
        metric_i=metric_dict[data_i][clf_type]
        sig_i=[]
        for clf_j in other_clfs:
            metric_j=metric_dict[data_i][clf_j]
            diff_i=np.mean(metric_i)-np.mean(metric_j)
            pvalue=stats.ttest_ind(metric_i,metric_j,
                               equal_var=False)[1]
            sign_ij= int(pvalue<0.05) * np.sign(diff_i)
            sig_i.append(sign_ij)
        sig_matrix.append(sig_i)
    sig_matrix=np.array(sig_matrix)
    title=f"Statistical significance ({clf_type}/{metric})"
    heatmap(matrix=sig_matrix,
            x_labels=other_clfs,
            y_labels=result_dict.data(),
            title=title)
    print(sig_matrix)

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

def pvalue_pairs(in_path,x_clf="RF",y_clf="MLP",metric="acc"):
    result_dict=eval.get_result_dict(in_path)
    x_dict=result_dict.get_clf(clf_type=x_clf,metric=metric)
    y_dict=result_dict.get_clf(clf_type=y_clf,metric=metric)
    def helper(data_i):
        line=[data_i]
        x_value=np.mean(x_dict[data_i])
        y_value=np.mean(y_dict[data_i])
        diff_i=x_value-y_value
        line+=[x_value,y_value,diff_i]
        pvalue=stats.ttest_ind(x_dict[data_i],y_dict[data_i],
                               equal_var=False)[1]
        line.append(pvalue)
        return line
    df=dataset.make_df(helper,
            iterable=result_dict.data(),
            cols=["data",x_clf,y_clf,"diff","pvalue"])
    df.df["sign"]=df.df.apply(lambda x: x.pvalue<0.05, axis=1)
    df.df["change"]=df.df.apply(lambda x: int(x.sign)*np.sign(x["diff"]), axis=1) 
    df.print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="binary_exp/exp")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--clf", type=str, default="RF")
    args = parser.parse_args()
    if(args.pair):
    	x_clf,y_clf=args.pair.split(",")
    	pvalue_pairs(in_path=args.data,
    		         x_clf=x_clf,
    		         y_clf=y_clf,
    		         metric=args.metric)
    else:
    	pvalue_matrix(in_path=args.data,
    		          clf_type=args.clf,
    		          metric=args.metric)