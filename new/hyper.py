import numpy as np
import pandas as pd
from itertools import product
import os.path
import base,dataset,tree_clf,utils

class HyperFile(object):
    def __init__(self,file_path):
        self.file_path=file_path
        self.metrics=["acc","balance"]

    def add_param(self,data_id,param_dict,result):
        line=[data_id]
        names=list(param_dict.keys())
        names.sort()
        if(not os.path.exists(self.file_path)):
            header=["data"]+names+self.metrics
            self.save_line(header)
        for name_i in names:
            param_i=param_dict[name_i]
            line.append(str(param_i))
        for metric_i in self.metrics:
            metric_value_i=result.get_metric(metric_i)
            line.append(f"{np.mean(metric_value_i):.4f}")
        print(line)
        self.save_line(line)

    def save_line(self,line):
        with open(self.file_path, 'a') as file:
             file.write(",".join(line)+"\n")

class HyperparamSpace(object):
    def __init__( self,
                  extr=["info","ind","prod"],
                  n_feats=[10,20,30,50]):
        self.extr=extr
        self.n_feats=n_feats

    def __call__(self):
        for feat_i,dim_i in product(self.extr,self.n_feats):
            yield { "feat_type":feat_i,
                      "n_feats":dim_i}


def optim_hyper(in_path,out_path):
    data_id=in_path.split("/")[-1]
    data=dataset.read_csv(in_path)
    splits=base.get_splits(data,10,3)
    hyper_file=HyperFile(out_path)
    hyper_space=HyperparamSpace()
    for hyper_i in hyper_space():
        factory_i=tree_clf.TreeFactory(hyper_i)
        result_i=factory_i.get_results(data,
                                      enumerate(splits))
        hyper_file.add_param(data_id,hyper_i,result_i)


def optim_exp(paths,hyper_path):
    all_paths=[]
    for path_i in paths:
        all_paths+=utils.top_files(path_i)
    for path_i in all_paths:
        optim_hyper(path_i,hyper_path)


def read_hyper(in_path):
    df=pd.read_csv(in_path)
    df=dataset.DFView(df)
    best_df=dataset.DFView(df.best())
    feat_dict=best_df.get_dict("data","feat_type")
    dim_dict=best_df.get_dict("data","n_feats")
    hyper_dict={}
    for key_i in feat_dict.keys():
        hyper_dict[key_i]={ "feat_type":feat_dict[key_i],
                            "n_feats":dim_dict[key_i]
                           }
    print(hyper_dict)
    return hyper_dict

read_hyper("hyper_full.csv")
#in_path="../incr_exp/uci/data/vehicle"
#optim_hyper(in_path,"hyper.csv")
#optim_exp(["test/A","test/B"],"hyper_full.csv")
