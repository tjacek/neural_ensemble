import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import itertools,os.path
import dataset,pred,plot,utils

def metric_plot(conf_dict):
    metric,text=conf_dict["metric"],conf_dict["text"]
    x_clf,y_clf=conf_dict["x"],conf_dict["y"]
    result_dict=pred.unify_results(conf_dict["result"])
    x_dict=result_dict.get_mean_metric(x_clf,metric=metric)
    y_dict=result_dict.get_mean_metric(y_clf,metric=metric)
    if("names" in conf_dict):
        text=conf_dict["names"]
    plot.dict_plot( x_dict,
                    y_dict,
                    xlabel=f"{x_clf}({metric})",
                    ylabel=f"{y_clf}({metric})",
                    text=text,
                    title=conf_dict["title"])

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

class HyperSource(object):
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
        if(dim_i=="?"):
            return best_dict(self.df,feat_i,dim_i,metric)
        return selected_dict(self.df,feat_i,dim_i,metric)

    def __contains__(self, item):
        feat_i,dim_i=item
        if(not feat_i in self.feats):
            return False
        if(not dim_i in self.dims and 
            dim_i!='?'):
            return False
        return True

class CsvSource(object):
    def __init__(self,df):
        self.df=df
        self.clfs=set(df["clf"].tolist())
    
    def __call__(self,clf_type,metric):
        df_clf=self.df[self.df["clf"]==clf_type]
        return dict(zip(df_clf['data'].tolist(),
                        df_clf[metric].tolist()))
    def __contains__(self, item):
        return item in self.clfs

def parse_csv(in_path):
    df=pd.read_csv(in_path)
    cols=set(df.columns)
    if("clf" in cols):
        return CsvSource(df)
    return HyperSource(df)

def selected_dict(df,feat_i,dim_i,metric):
    df=df[df['feats']==f"'{feat_i}'"]
    df=df[df['dims']==dim_i]
    df=df[['data',metric]]
    return dict(zip(df['data'].tolist(),
                      df[metric].tolist()))

def best_dict(df,feat_i,dim_i,metric):
    grouped=df.groupby(by='data')
    def helper(df_i):
        df_i=df_i.sort_values(by=metric,ascending=False)
        row_j=df_i.iloc[0]
        return row_j     
    df=grouped.apply(helper)
    return dict(zip(df['data'].tolist(),
                      df[metric].tolist()))

def make_source(result,metric):
    data_sources=[]
    for path_i in result:
        if(os.path.isdir(path_i)):
            result_dict=pred.unify_results([path_i])
            source_i=DirSource(result_dict)
        else:
            source_i=parse_csv(path_i)
        data_sources.append(source_i)
    def helper(clf_type):
        clf_dict={}
        for source_i in data_sources:
            if(clf_type in source_i):
                clf_dict=clf_dict | source_i(clf_type,metric)
        return clf_dict
#        raise Exception(f"Unknown clf type {clf_type}")
    return helper

def hete_sources(conf_dict):
    result,metric=conf_dict["result"],conf_dict['metric']
    helper=make_source(result,metric)
    x_clf,y_clf=conf_dict['x'],conf_dict['y']
    x_dict= helper(x_clf)
    y_dict= helper(y_clf)
    print(x_dict)
    print(y_dict)
    plot.dict_plot( x_dict,
                    y_dict,
                    xlabel=f"{x_clf}({metric})",
                    ylabel=f"{y_clf}({metric})",
                    text=conf_dict["text"])

def diff(conf_dict):
    result,metric=conf_dict["result"],conf_dict['metric']
    helper=make_source(result,metric)
    desc_dict= get_desc(conf_dict['data'],conf_dict['desc'])
    desc_dict={ name_i:float(value_i) 
            for name_i,value_i in desc_dict.items()}
    x,y=conf_dict['x'],conf_dict['y']
    x_dict= helper(x)
    y_dict= helper(y)
    print(x_dict.keys())
    print(y_dict.keys())
    diff_dict={ key_i:(x_dict[key_i]-y_dict[key_i])
                            for key_i in x_dict}
    plot.dict_plot( desc_dict,
                    diff_dict,
                    xlabel=conf_dict["desc"],
                    ylabel=f"{x}-{y}",
                    text=conf_dict["text"],
                    title=conf_dict["title"])

def get_desc(in_path,desc):
    if(type(in_path)==list):
        full_dict={}
        for path_i in in_path:
            full_dict=full_dict|get_desc(path_i,desc)
        return full_dict
    else:
        df=dataset.csv_desc(in_path)
        desc_dict= df.get_dict("id",desc)
        return desc_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="hete.json")
    args = parser.parse_args()
    conf_dict=utils.read_json(args.conf)
    if(conf_dict["type"]=="diff"):
        diff(conf_dict)
    else:
        metric_plot(conf_dict)
#    feat_hist("binary_exp/exp")