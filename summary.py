import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import itertools,os.path
import pred,plot,utils

def metric_plot(conf_dict):
    metric,text=conf_dict["metric"],conf_dict["text"]
    x_clf,y_clf=conf_dict["x_clf"],conf_dict["y_clf"]
    result_dict=pred.unify_results(conf_dict["exp_path"])
    x_dict=result_dict.get_mean_metric(x_clf,metric=metric)
    y_dict=result_dict.get_mean_metric(y_clf,metric=metric)
    if("names" in conf_dict):
        text=conf_dict["names"]
    plot.dict_plot( x_dict,
                    y_dict,
                    xlabel=f"{x_clf}({metric})",
                    ylabel=f"{y_clf}({metric})",
                    text=text)

def count_feats(conf_dict):
    @utils.DirFun(out_arg=None)
    def helper(in_path):
        model_path=f"{in_path}/TREE-MLP/models"
        n_feats=[ feat_i.shape[0] 
                  for feat_i in iter_feats(model_path)] 
        return np.mean(n_feats),np.std(n_feats)
    for exp_i in conf_dict['exp_path']:
        output=helper(exp_i)
        for name_i,value_i in output.items():
            print(f"{name_i},{value_i}")

def feat_hist(in_path):
    @utils.DirFun(out_arg=None)
    def helper(in_path):
        feat_dict=defaultdict(lambda:0)
        model_path=f"{in_path}/TREE-MLP/models"
        for feat_i in iter_feats(model_path):
            for feat_j  in feat_i:
                feat_dict[feat_j]+=1
        keys=list(feat_dict.keys())
        keys.sort()
        y=[feat_dict[key_i] for key_i in keys]
        plt.bar(keys,y)
        plt.title(in_path.split("/")[-1])
        plt.show()
    helper(in_path)    

def iter_feats(model_path):
    for model_i in utils.top_files(model_path):
        feat_path_i=f"{model_i}/tree/feats.npy"
        feat_i=np.load(feat_path_i)
        yield feat_i

class DirSource(object):
    def __init__(self,result_dict):
        self.result_dict=result_dict
        self.clfs= set(result_dict.clfs())
    
    def __call__(self,clf_type,metric):
        clf_dict=self.result_dict.get_clf(clf_type,metric)
        return { clf_i:np.mean(value_i) 
                    for clf_i,value_i in clf_dict.items()}

    def __contains__(self, item):
        if(type(item)==list):
            return False
        return item in self.clfs 

class FileSource(object):
    def __init__(self,df):
        self.df=df[['data', 'feats', 'dims',
                    'accuracy','balance']]
        self.feats=set([feat_i.replace("'","") 
                        for feat_i in df['feats'].unique()])
        self.dims=set(df['dims'].unique())
    
    def __call__(self,clf_type,metric):
        if(type(clf_type)!=list):
            return False
        feat_i,dim_i=clf_type
        grouped=self.df.groupby(by='data')
        df=self.df[ self.df['feats']==f"'{feat_i}'"]
        df=df[df['dims']==dim_i]
        df=df[['data',metric]]
        clf_dict=dict(zip(df['data'].tolist(),
                          df[metric].tolist()))
        return clf_dict

    def __contains__(self, item):
        feat_i,dim_i=item
        if(not feat_i in self.feats):
            return False
        if(not dim_i in self.dims):
            return False
        return True

def diff_sources(conf_dict):
    print(conf_dict)
    data_sources=[]
    for path_i in conf_dict["data"]:
        if(os.path.isdir(path_i)):
            result_dict=pred.unify_results([path_i])
            source_i=DirSource(result_dict)
        else:
            df=pd.read_csv(path_i)
            source_i=FileSource(df)
        data_sources.append(source_i)
    metric=conf_dict['metric']
    def helper(clf_type):
        for source_i in data_sources:
            if(clf_type in source_i):
                return source_i(clf_type,metric)
        raise Exception(f"Unknown clf type {clf_type}")
        print(clf_type)
    x_clf,y_clf=conf_dict['x'],conf_dict['y']
    x_dict= helper(x_clf)
    y_dict= helper(y_clf)
    plot.dict_plot( x_dict,
                    y_dict,
                    xlabel=f"{x_clf}({metric})",
                    ylabel=f"{y_clf}({metric})",
                    text=conf_dict["text"])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="hete.json")
    args = parser.parse_args()
    conf_dict=utils.read_json(args.conf)
    diff_sources(conf_dict)
#    metric_plot(conf_dict)
#    feat_hist("binary_exp/exp")