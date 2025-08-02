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
        tree_factory=tree_clf.TreeFeatFactory(clf_factory)
        return str(tree_factory),tree_factory
    desc=str(clf_factory)
    return desc,clf_factory

def prepare_clfs(clf_descs,proto):
    new_clfs=[]
    for desc_i in clfs_desc:
        if(type(desc_i)==dict):
            new_clfs+=from_desc(desc_i,proto)
        else:
            new_clfs.append(desc_i)
    return new_clfs

def from_desc(desc,proto):
    variants=[proto.copy()]
    keys=[ key_j
            for key_j in desc
                if(not key_j=="type")]
    for key_i in keys:
        new_variants=[]
        for arg_j in desc[key_i]:
            for var_k in variants:
                var_ijk=var_k.copy()
                var_ijk[key_i]=arg_j
                new_variants.append(var_ijk) 
        variants=new_variants
    for var_i in variants:
        if("n_feats" in var_i):
            n_feats=var_i["n_feats"]
            extr=var_i["extr_factory"]
            var_i["extr_factory"]=(extr,n_feats)
            del var_i["n_feats"]
    variants=[tree_clf.TreeFeatFactory(var_i,desc["type"]) 
                 for var_i in variants]
    return variants
      
prototype={ "tree_factory":"random",
            "clf_type":"SVM"}

clfs_desc=[tree_feats.RandomTree(),
           { "type":"ens",
             "extr_factory":["info","ind"],
             "n_feats":[20,30,50],
             "concat":[True,False]},
            "SVM"]

clfs=prepare_clfs(clfs_desc,prototype)
compare_exp(in_path="bad_exp/data/wine-quality-red",
	        clfs=clfs)