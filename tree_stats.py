import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import base,clfs,dataset

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

def get_raw_trees(clf):
    clf_type=get_type(clf)
    if(clf_type=="GRAD"):
        raw=[est_j.tree_
                for est_i in clf.estimators_
                    for est_j in est_i] 
    else:	
        raw=[est_i.tree_ for est_i in clf.estimators_]
    return raw

def tree_histogram(clf):
    n_feats=clf.n_features_in_
    all_hist=[]
    for tree_i in get_raw_trees(clf):
        hist_i=np.zeros((n_feats))
        for feat_j in tree_i.feature:
            if(feat_j >=0):
                hist_i[feat_j]+=1
        all_hist.append(hist_i)
    all_hist=np.array(all_hist)
    feat_hist=np.sum(all_hist,axis=0)
    print(feat_hist/ np.sum(feat_hist))
#        print(est_i.features)

clf=get_clf("bad_exp/data/wine-quality-red",
	        clf_type="RF")
tree_histogram(clf)