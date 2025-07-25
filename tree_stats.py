import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import base,clfs,dataset,utils

def get_clf(in_path,clf_type="RF"):
    data_i=dataset.read_csv(in_path)
    split_k=base.random_split(data_i)
    clf=base.get_clf(clf_type=clf_type)
    result,_=split_k.eval(data_i,clf)
    print(result.get_acc())
    return clf

def get_type(clf):
    if(type(clf)==RandomForestClassifier):
        return "RF"
    elif(type(clf)==GradientBoostingClassifier):
        return "GRAD"	

def get_indiv_trees(clf):
    clf_type=get_type(clf)
    if(clf_type=="GRAD"):
        trees=[]
        for est_i in clf.estimators_:
            trees+=list(est_i)
        raise Exception(trees[0]==trees[-1])
        return trees
    else:	
        return clf.estimators_

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
#        print(est_i.features)

def tree_depths(clf):
    depths=[[tree_i.tree_.max_depth,
             tree_i.tree_.node_count] 
                for tree_i in get_indiv_trees(clf)]
    print(depths)

def tree_acc(in_path,clf_type="RF"):
    data=dataset.read_csv(in_path)
    split=base.random_split(data)
    clf=base.get_clf(clf_type=clf_type)
    result,_=split.eval(data,clf)
    acc=[]
    indv_trees=get_indiv_trees(clf)
    for tree_j in tqdm(indv_trees):
        result_j,_=split.eval(data,tree_j)
        result_j.y_pred= result_j.y_pred.astype(int)
#        raise Exception(result_j.y_true ,result_j.y_pred)
        acc.append(result_j.get_acc())
    acc=np.array(acc)
    print(acc)
#    print((acc-np.mean(acc))/np.std(acc))

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

clf=tree_acc("bad_exp/data/wine-quality-red",
	        clf_type="GRAD")
#tree_histogram(clf)
#stats("uci")