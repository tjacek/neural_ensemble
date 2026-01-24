import numpy as np
import pandas as pd
from itertools import product
import os.path
import base,dataset,tree_clf,utils

class BestHyper(dict):
    def __init__(self, arg=[]):
        super(BestHyper, self).__init__(arg)

    def save(self,out_path):
        params=list(self.values())[0].keys()
        params=list(params)
        params.sort()
        with open(out_path, 'a') as file:
            file.write(",".join(["data"]+params)+"\n")
            for name_i,dict_i in self.items():
                raw_i=[name_i]
                for key_i in params:
                    raw_i.append(str(dict_i[key_i]))
                file.write(",".join(raw_i)+"\n")

    @classmethod
    def read(cls,in_path:str):
        df=pd.read_csv(in_path)
        df=dataset.DFView(df)
        best_hyper=cls()
        for data_i in df.by_data(None):
            dict_i=data_i.to_dict()
            dict_i={ key_i:list(value_i.values())[0]
                      for key_i,value_i in dict_i.items()}
            name_i=dict_i["data"]
            del dict_i["data"]
            best_hyper[name_i]=dict_i
#        raise Exception(dir(df.df))
#        best_df=dataset.DFView(df.best())
#        feat_dict=best_df.get_dict("data","feat_type")
#        dim_dict=best_df.get_dict("data","n_feats")
#        hyper_dict={}
#        for key_i in feat_dict.keys():
#            hyper_dict[key_i]={ "feat_type":feat_dict[key_i],
#                                "n_feats":dim_dict[key_i]
#                              }
        return best_hyper

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

class HyperFull(dict):
    def __init__(self, arg=[]):
        super(HyperFull, self).__init__(arg)
    
    def diff(self,sort="acc"):
        for name_i,df_i in self.items():
            df_i=df_i.sort_values(by=sort,ascending=False)
            metric=df_i[sort].tolist()
            diff=100*(metric[0]-metric[-1])
            print(f"{name_i}:{diff:.2f}%")

    @staticmethod
    def read(in_path):
        df=pd.read_csv(in_path)
        df=dataset.DFView(df)
        hyper_full=HyperFull()
        for df_i in df.by_data(sort="acc"):
            name_i=df_i['data'].tolist()[0]
            hyper_full[name_i]=df_i
        return hyper_full

class HyperparamSpace(object):
    param_names=["feat_type","n_feats"]
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
        factory_i=tree_clf.TreeFactory
        result_i=factory_i.get_results(data,
                                      enumerate(splits),
                                      hyper_i)
        hyper_file.add_param(data_id,hyper_i,result_i)

    return data_id,splits

def optim_exp(paths,hyper_path,split_path):
    all_paths=[]
    for path_i in paths:
        all_paths+=utils.top_files(path_i)
    utils.make_dir(split_path)
    for path_i in all_paths:
        data_id,splits=optim_hyper(path_i,hyper_path)
        split_path_i=f"{split_path}/{data_id}"
        utils.make_dir(split_path_i)
        split_path_i=f"{split_path_i}/splits"
        utils.make_dir(split_path_i)
        base.save_splits(split_path_i,splits)

if __name__ == '__main__':
#    hyper_full=HyperFull.read("hyper_good.csv")
#    hyper_full.diff()
#read_hyper("hyper_full.csv")
#in_path="../incr_exp/uci/data/vehicle"
#optim_hyper(in_path,"hyper.csv")
    optim_exp(["test/A","test/B"],
               "hyper_goodII.csv",
               "test_exp")
