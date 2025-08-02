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

#def clfs_desc(extr_types:list,
#              feat_sizes:list,
#              clf_type="SVM",
#              tree="random",):
#    clfs=["SVM",tree_feats.RandomTree()]
#    for extr_i in extr_types:
#        for feat_i in feat_sizes:
#            clf_i={ "tree_factory":tree,
#                    "extr_factory":(extr_i,feat_i),
#                    "clf_type":clf_type}
#            clfs.append(clf_i)
#    return prepare_clfs(clfs)

def prepare_clfs(clf_descs,proto):
#    taboo=set(["type"]+list(proto.keys()))
    new_clfs=[]
    for desc_i in clfs_desc:
        if(type(desc_i)==dict):
            new_clfs+=from_desc(desc_i,proto)
#            conc_clf_i=clf_i.copy()
#            conc_clf_i["concat"]=True
#            new_clfs.append(conc_clf_i) 
#            clf_i["concat"]=False
#            new_clfs.append(clf_i)
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
    return variants
      
prototype={ "tree_factory":"random",
            "clf_type":"SVM"}

clfs_desc=[tree_feats.RandomTree(),
           { "type":"clf",
             "extr_factory":["cs","info"],
#             "n_feats":[20,30],
             "concat":[True,False]},
            "SVM"]

clfs=prepare_clfs(clfs_desc,prototype)
print(clfs)
#clfs=[ tree_feats.GradientTree(),
#       tree_feats.RandomTree(),
#       { "tree_factory":"random",
#         "extr_factory":("cs",20),
#         "clf_type":"SVM",
#         "ens":True},
#       { "tree_factory":"random",
#         "extr_factory":("info",30),
#         "clf_type":"SVM"},
#       "LR",
#       "SVM"]
#clfs=clfs_desc(extr_types=["info","disc"],
#              feat_sizes=[20,30,40,50])
compare_exp(in_path="bad_exp/data/wine-quality-red",
	        clfs=clfs)