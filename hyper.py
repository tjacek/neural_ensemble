import numpy as np
#from sklearn.decomposition import PCA
from tqdm import tqdm
from itertools import product
import argparse
import base,clfs,utils,tree_clf,tree_feats,dataset
utils.silence_warnings()

class ParamSpace(object):
    def __init__( self,
                  extr=["info","ind","prod"],
                  n_feats=[10,20,30,50],
                  units=[1,2]):
        self.extr=extr
        self.n_feats=n_feats
        self.units=units

    def iter(self):
        for feat_i,dim_i in product(extr,n_feats):
            arg_i={ "tree_factory":"random",
                    "extr_factory":(feat_i,dim_i),
                    "concat":True}
            for unit_j in units:
                hyper_j={ 'layers':2, 'units_0':unit_j,
                          'units_1':1,'batch':False}
                yield arg_i,hyper_j

def nn_tree(in_path,multi=False,selected=None):    
    if(selected):
        selected=set(selected)
    extr=["info","ind","prod"]
    n_feats=[10,20,30,50]
    units=[1,2]
    def arg_iter():
        for feat_i,dim_i in product(extr,n_feats):
            arg_i={ "tree_factory":"random",
                    "extr_factory":(feat_i,dim_i),
                    "concat":True}
            for unit_j in units:
                hyper_j={ 'layers':2, 'units_0':unit_j,
                          'units_1':1,'batch':False}
                yield arg_i,hyper_j
    def helper(in_path):
        data=in_path.split("/")[-1]
        print(data)
        if(selected and not (data in selected)):
            return []
        data_split=base.get_splits( data_path=in_path,
                                    n_splits=10,
                                    n_repeats=1)
        lines=[]
        for arg_i,hyper_i in arg_iter():
            feat_i,dim_i=arg_i['extr_factory']
            clf_factory=clfs.get_clfs("TREE-MLP",
                                      hyper_params=hyper_i,
                                      feature_params=arg_i)
            clf_factory.init(data_split.data)
            results=data_split.get_results(clf_factory)
            acc_i=np.mean(results.get_metric("acc"))
            balance_i=np.mean(results.get_metric("balance"))
            line_i=[feat_i,dim_i,hyper_i['units_0']]
            desc_i=",".join([str(c_j) for c_j in line_i])
            print(f"{data},{desc_i},{acc_i:.4f},{balance_i:.4f}")
            line_i= [data] + line_i + [acc_i,balance_i]
            lines.append(line_i)
        return lines
    cols=[ "data","feats","dims","layer_1",
           "acc","balance"]
    if(multi):
        df=dataset.make_df(helper=helper,
                           iterable=utils.top_files(in_path),
                           cols=cols,
                           offset=None,
                           multi=True)
    else:
        lines=helper(in_path)
        df=dataset.from_lines(lines,cols)
    df.by_data()
    return df

def svm_tree(in_path,multi=False,selected=None):
    if(selected):
        selected=set(selected)
    extr=["info","ind","prod"]
    n_feats=[10,20,30,50]
    def helper(in_path):
        data=in_path.split("/")[-1]
        if(selected and not (data in selected)):
            raise Exception(data)
        data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
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
    df.by_data()
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str,  default="uci")
    parser.add_argument("--out_path", type=str, default="uci_out")
    parser.add_argument("--hyper_path", type=str, default="uci_neural.csv")
    args = parser.parse_args()
    print(args)
    df=nn_tree(f"{args.in_path}_exp/data",multi=True,
                  selected=[ "mfeat-fourier","mfeat-karh",
                             "newthyroid", "satimage"])
    if(args.out_path):
        df.df.to_csv(args.out_path, sep=',')
    if(args.hyper_path):
        best_df=df.best()[['data','feats','dims','acc']]
#        print(best_df)
        best_df.to_csv(args.hyper_path, sep=',')