import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import base,clfs,dataset,plot,utils

class TreeClfAdapter(object):
    def __init__(self,clf,split,data):
        self.clf=clf
        self.split=split
        self.data=data

    def get_indiv_trees(self):
        if(str(self)=="GRAD"):
            trees=[]
            for est_i in self.clf.estimators_:
                trees+=list(est_i)
            return trees
        else:   
            return self.clf.estimators_

    def __str__(self):
        if(type(self.clf)==RandomForestClassifier):
            return "RF"
        elif(type(self.clf)==GradientBoostingClassifier):
            return "GRAD"   

    def __repr__(self):
        return str(self)

def tree_acc(in_path,
             clf_type="RF",
             metric_type="acc"):
    clf=get_clf(in_path,clf_type=clf_type)
    acc=[]
    indv_trees=clf.get_indiv_trees()
    print(len(indv_trees))
    for tree_j in tqdm(indv_trees):
        result_j,_= clf.split.eval(clf.data,tree_j)
        result_j.y_pred= result_j.y_pred.astype(int)
        acc.append(result_j.get_metric(metric_type))
    print(np.mean(acc))
    print(np.std(acc))
    return np.array(acc)

def get_clf(in_path,clf_type="RF"):
    data_i=dataset.read_csv(in_path)
    split_k=base.random_split(data_i,p=0.9)
    clf=base.get_clf(clf_type=clf_type)
    result,_=split_k.eval(data_i,clf)
    print(result.get_acc())
    return TreeClfAdapter(clf,split_k,data_i)

def tree_histogram(clf):
    n_feats=clf.n_features_in_
    all_hist=[]
    for tree_i in get_indiv_trees(clf):
        hist_i=np.zeros((n_feats))
        for feat_j in tree_i.tree_.feature:
            if(feat_j >=0):
                hist_i[feat_j]+=1
        all_hist.append(hist_i)
    all_hist=np.array(all_hist)
    feat_hist=np.sum(all_hist,axis=0)
    return (feat_hist/ np.sum(feat_hist))

def tree_depths(clf):
    depths=[[tree_i.tree_.max_depth,
             tree_i.tree_.node_count] 
                for tree_i in get_indiv_trees(clf)]
    print(depths)

def show_acc(in_path,metric_type="acc"):
    for path_i in utils.top_files(in_path):
        acc_i=tree_acc(path_i,
                       clf_type="RF",
                       metric_type=metric_type)
        x,dens= plot.compute_density(acc_i)
        plot.simple_plot(x=x,
                         y=dens,
                         title=path_i.split("/")[-1],
                         xlabel=metric_type)


def show_param(in_path,step=5,max_clf=100):
    n_iters= int(max_clf/step)
    x=[ (j+1)*step for j in range(n_iters)] 
    for path_i in utils.top_files(in_path):
        data_i=dataset.read_csv(path_i)
        split_i=base.random_split(data_i)
        acc=[]
        for x_j in x:
            clf_j=RandomForestClassifier(n_estimators=x_j)
            result_j,_=split_i.eval(data_i,clf_j)
            acc.append(result_j.get_metric("acc"))
        plot.simple_plot(x=x,
                         y=acc,
                         title=path_i.split("/")[-1],
                         xlabel="acc")

def stats(in_path):
    @utils.DirFun(out_arg=None)
    def helper(in_path):
        print(in_path)
        clf_i=get_clf(in_path,clf_type="RF")
        hist_i=tree_histogram(clf_i)
        return hoover_index(hist_i)
    output=helper(in_path)
    print(output)

def hoover_index(x):
    diff=np.abs(x-np.mean(x))
    return 0.5*np.sum(diff)/np.sum(x)

if __name__ == '__main__':
    in_path="neural/uci/data/"#yeast"
    show_acc(in_path)