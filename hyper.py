import numpy as np
#from sklearn.decomposition import PCA
from tqdm import tqdm
from itertools import product
import base,clfs,utils,tree_clf,tree_feats,dataset
utils.silence_warnings()

def hyper_comp(in_path,hyper,clf_type="TREE-ENS"):
    data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
    feature_params={"tree_factory":"random",
                            "extr_factory":("info",30),
                            "concat":True}
    for hyper_i in hyper:
        clf_factory=clfs.get_clfs(clf_type,
                                 hyper_params=hyper_i,
                                 feature_params=None)
        eval_tree(data_split,clf_factory)

def tree_comp(in_path,
              clf_type="TREE-ENS",
              extr=None,
              n_feats=None):
    data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
    prototype={"tree_factory":"random",
                "concat":True}
    if(extr is None):
        extr=["info","ind"]
    if(n_feats is None):
        n_feats=[20,30,50]
    hyper={'layers':2, 'units_0':1,'units_1':1,'batch':False}
    for feat_i,dim_i in product(extr,n_feats):
        tree_i=prototype.copy()
        tree_i["extr_factory"]= (feat_i,dim_i)#extr_i
        clf_factory=clfs.get_clfs(clf_type,
                                 hyper_params=hyper,
                                 feature_params=tree_i)
        acc_i,balance_i=eval_tree(data_split,clf_factory)
        print(f"{feat_i}.{dim_i},{acc_i:.4f},{balance_i:.4f}")

def eval_tree(data_split,clf_factory):
    clf_factory.init(data_split.data)
    acc,balance=[],[]
    for clf_j,result_j in tqdm(data_split.eval(clf_factory)):
        acc.append(result_j.get_acc())
        balance.append(result_j.get_metric("balance"))
    return np.mean(acc),np.mean(balance)


def svm_tree(in_path,multi=False):
    extr=["info","ind"]
    n_feats=[10,20,30,50]
#    @utils.DirFun("in_path")#,"exp_path")
    def helper(in_path):
        data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
        data=in_path.split("/")[-1]
        lines=[]
        for feat_i,dim_i in product(extr,n_feats):
            arg_i={ "tree_factory":"random",
                 "extr_factory":(feat_i,dim_i),
                 "clf_type":"SVM",
                 "concat":True}
            factory_i=tree_clf.TreeFeatFactory(arg_i)
            results=data_split.get_results(factory_i)
            acc_i=np.mean(results.get_metric("acc"))
            balance_i=np.mean(results.get_metric("balance"))
            desc_i=str(factory_i)
            print(f"{data},{desc_i},{acc_i:.4f},{balance_i:.4f}")
            line_i=desc_i.split(",")[1:]
            line_i= [data] + line_i + [acc_i,balance_i]
            lines.append(line_i)
        return lines
    cols=[ "data","clf","concat","feats",
           "dims","tree","acc","balance"]
    if(multi):
        df=dataset.make_df(helper=helper,
                           iterable=utils.top_files(in_path),
                           cols=cols,
                           offset=None,
                           multi=True)
    else:
        lines=helper(in_path)
        df=dataset.from_lines(lines,cols)
    df.print()
#    print(lines)

if __name__ == '__main__':
    hyper=[{'layers':2, 'units_0':2,'units_1':1,'batch':False}]#,
    in_path="binary_exp/data" #"-quality-red"
#    in_path="multi_exp/data/first-order"
#    tree_comp( in_path,
#               clf_type="TREE-ENS",
#               extr=["mixed"],
#               n_feats=[30])
    svm_tree(in_path,multi=True)