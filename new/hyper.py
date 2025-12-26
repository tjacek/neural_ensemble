import numpy as np
from itertools import product
import base,dataset,tree_clf

class HyperFile(object):
    def __init__(self,file_path):
        self.file_path=file_path

    def add_line(self,data_id,param_dict,result):
        line=[data_id]
        names=list(param_dict.keys())
        names.sort()
        for name_i in names:
            param_i=param_dict[name_i]
            line.append(str(param_i))
        for metric_i in ["acc","balance"]:
            metric_value_i=result.get_metric(metric_i)
            line.append(f"{np.mean(metric_value_i):.4f}")
        print(line)
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
    splits=base.get_splits(data,10,1)
    hyper_file=HyperFile(out_path)
    hyper_space=HyperparamSpace()
    for hyper_i in hyper_space():
        factory_i=tree_clf.TreeFactory(hyper_i)
        result_i=factory_i.get_results(data,
                                      enumerate(splits))
        hyper_file.add_line(data_id,hyper_i,result_i)

in_path="../incr_exp/uci/data/vehicle"
optim_hyper(in_path,"hyper.csv")