import numpy as np
#from sklearn.decomposition import PCA
from tqdm import tqdm
from itertools import product
import base,clfs,utils#dataset,tree_feats
utils.silence_warnings()

def hyper_comp(in_path,hyper):
    data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
    tree_i={ "clf_factory":"random",
             "extr_factory":("info",30),
             "concat":True,
             "ens_type":"binary",
             "weights":"specific"}
    for hyper_i in hyper:
        eval_tree(data_split,hyper_i,tree_i)

def tree_comp(in_path):
    data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
    prototype={"clf_factory":"random",
                "concat":True,
                "ens_type":"full",
                "weights":"specific"}
    extr=["info","ind"]
    n_feats=[20,30,50]
    hyper={'layers':2, 'units_0':1,'units_1':1,'batch':False}
    for extr_i in product(extr,n_feats):
        tree_i=prototype.copy()
        tree_i["extr_factory"]=extr_i
        eval_tree(data_split,hyper,tree_i)

def eval_tree(data_split,hyper_i,tree_i):
    nn_factory_i=clfs.CSTreeEnsFactory(hyper_i,tree_i)
    nn_factory_i.init(data_split.data)
    acc_i,balance_i=[],[]
    for clf_j,result_j in tqdm(data_split.eval(nn_factory_i)):
        acc_i.append(result_j.get_acc())
        balance_i.append(result_j.get_metric("balance"))
    print(tree_i)
    print(hyper_i)
    print(f"{np.mean(acc_i):.4f},{np.mean(balance_i):.4f}")


if __name__ == '__main__':
    hyper=[{'layers':2, 'units_0':2,'units_1':1,'batch':False}]#,
#           {'layers':2, 'units_0':1,'units_1':1,'batch':False}]#,
#           {'layers':2, 'units_0':2,'units_1':1,'batch':True},
#           {'layers':2, 'units_0':1,'units_1':1,'batch':True},]
    in_path="bad_exp/data/wine-quality-red"
    hyper_comp(in_path,hyper)
#    tree_comp(in_path)