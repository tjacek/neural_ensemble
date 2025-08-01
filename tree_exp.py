import numpy as np
import base
import tree_feats,tree_clf,utils
utils.silence_warnings()

def compare_exp(in_path,clfs):
    data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
    for clf_type_i in clfs:
        desc_i,clf_factory_i=get_clf(clf_type_i)
        acc_i,balance_i=[],[]
        for clf_j,result_j in data_split.eval(clf_factory_i):
            acc_i.append(result_j.get_acc())
            balance_i.append(result_j.get_metric("balance"))
        print(f"{desc_i}:{np.mean(acc_i):.4f},{np.mean(balance_i):.4f}")


def get_clf(clf_factory):
    if(type(clf_factory)==str):
        desc=clf_factory
        clf_factory=base.ClasicalClfFactory(clf_factory)
        return desc,clf_factory
    if(type(clf_factory)==dict):
        names=list(clf_factory.keys())
        names.sort()
        desc=[str(clf_factory[name_i]) 
                for name_i in names]
        desc=",".join(desc)
        tree_factory=tree_clf.TreeFeatFactory(clf_factory)
        return desc,tree_factory
    desc=str(clf_factory)
    return desc,clf_factory

def prepare_clfs(clfs):
    new_clfs=[]
    for clf_i in clfs:
        if(type(clf_i)==dict):
            conc_clf_i=clf_i.copy()
            conc_clf_i["concat"]=True
            new_clfs.append(conc_clf_i) 
            clf_i["concat"]=False
            new_clfs.append(clf_i)
        else:
            new_clfs.append(clf_i)
    return new_clfs

clfs=[ tree_feats.GradientTree(),
       tree_feats.RandomTree(),
       { "tree_factory":"random",
         "extr_factory":"cs",
         "clf_type":"SVM"},
       { "tree_factory":"random",
         "extr_factory":"info",
         "clf_type":"SVM"},
       "LR",
       "SVM"]
clfs=prepare_clfs(clfs)
compare_exp(in_path="bad_exp/data/wine-quality-red",
	        clfs=clfs)