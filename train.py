import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
import os.path
import pandas as pd
import argparse,json
import base,clfs,dataset,utils
    

def train(data_path:str,
              out_path:str,
              clf_type="MLP",
              start=0,
              step=10,
              hyper_path=None):
    if(clf_type in base.NEURAL_CLFS):
        train_fun=neural_train
    else:
        train_fun=clf_train
    interval=base.Interval(start,step)
    train_fun(data_path=data_path,
              out_path=out_path,
              clf_type=clf_type,
              interval=interval,
              hyper_path=hyper_path)

def clf_train(data_path:str,
               out_path:str,
               clf_type="TREE-MLP",
               interval=None,
               hyper_path=None):    
    @utils.ParallelDirFun()
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        dir_proxy=base.get_dir_path(out_path=exp_path,
                                    clf_type=clf_type)
        clf_factory=clfs.get_clfs(clf_type,hyper_params=None)
        path_dict=dir_proxy.path_dict(indexes=interval,
                                      key="results")
        clf_factory.init(data)
        print(path_dict['results'])
        dir_proxy.make_dir("results")
        for i,split_path_i in tqdm(enumerate(path_dict['splits'])):
            clf_i=clf_factory()
            split_i=base.read_split(split_path_i)
            split_i.fit_clf(data,clf_i)
            result_i=clf_i.eval(data,split_i)
            result_i.save(path_dict["results"][i])
        dir_proxy.save_info(clf_factory)
    helper(data_path,out_path)

def neural_train( data_path:str,
                  out_path:str,
                  clf_type="MLP",
                  interval=None,
                  hyper_path=None):
    hyper_dict=parse_hyper(hyper_path)
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        data_id=in_path.split("/")[-1]
        print(data_id)
        dir_proxy=base.get_dir_path(out_path=exp_path,
                                    clf_type=clf_type)
        hyper_params,feature_params=hyper_dict[data_id]
        clf_factory=clfs.get_clfs(clf_type=clf_type,
        	                      hyper_params=hyper_params,
                                  feature_params=feature_params)
        clf_factory.init(data)
        dir_proxy.make_dir(["history","models","results"])
        path_dict=dir_proxy.path_dict(indexes=interval,
                                      key="models")
        for i,split_path_i in tqdm(enumerate(path_dict['splits'])):
            clf_i=clf_factory()
            split_i=base.read_split(split_path_i)
            clf_i,history_i=split_i.fit_clf(data,clf_i)
            history_i= history_to_dict(history_i)
            with open(f"{path_dict['history'][i]}", 'w') as f:
                json.dump(history_i, f)
            clf_i.save(path_dict["models"][i])
            result_i=clf_i.eval(data,split_i)
            result_i.save(path_dict["results"][i])
        dir_proxy.save_info(clf_factory)
    helper(data_path,out_path)            

def history_to_dict(history):
    if(type(history)==list):
        return [history_to_dict(history_i) for history_i in history]
    history=history.history
    key=list(history.keys())[0]
    hist_dict={'n_epochs':len(history[key])}
    for key_i in history.keys():
        hist_dict[key_i]=history[key_i][-1]
    return hist_dict

def parse_hyper(hyper_path):
    df=pd.read_csv(hyper_path)
    hyper_dict={}
    for index, row_i in df.iterrows():
        dict_i=row_i.to_dict()
        data_i=dict_i['data']
        extr_i=(dict_i["feats"].replace("'",""),dict_i["dims"])
        hyper_i={'layers':2, 'units_0':dict_i["layer"],
                 'units_1':1,'batch':False}
        hyper_dict[data_i]=[ hyper_i,
                             { "tree_factory":"random",
                               "extr_factory":extr_i,
                               "concat":True}]
    return hyper_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="neural/uci/data")
    parser.add_argument("--out_path", type=str, default="neural/uci/exp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--clf_type", type=str, default="MLP")
    parser.add_argument("--hyper_path", type=str, default="neural/uci/hyper.csv")

    args = parser.parse_args()
    print(args)
    train( data_path=args.data,
           out_path=args.out_path,
           clf_type=args.clf_type,
           start=args.start,
           step=args.step,
           hyper_path=args.hyper_path)