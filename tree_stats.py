import numpy as np
import base,clfs,dataset

def get_clf(in_path,clf_type="RF"):
    data_i=dataset.read_csv(in_path)
    split_k=base.random_split(data_i)
    clf=base.get_clf(clf_type=clf_type)
    result,_=split_k.eval(data_i,clf)
    print(result.get_acc())
    return clf

def tree_histogram(clf):
    n_feats=clf.n_features_in_
    for est_i in clf.estimators_:
        hist_i=np.zeros((n_feats))
        for feat_j in est_i.tree_.feature:#__getstate__()
            if(feat_j >=0):
                hist_i[feat_j]+=1
        print(hist_i)
#        print(est_i.features)

clf=get_clf("bad_exp/data/wine-quality-red",clf_type="RF")
tree_histogram(clf)