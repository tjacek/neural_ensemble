import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
import os.path
import argparse,json
import base,clfs,dataset,utils

def train(data_path:str,
              out_path:str,
              clf_type="MLP",
              start=0,
              step=10,
              retrain=False):
    if(clf_type in base.NEURAL_CLFS):
        train_fun=nn_train
    else:
        train_fun=clf_train
    interval=base.Interval(start,step)
    train_fun(data_path=data_path,
              out_path=out_path,
              clf_type=clf_type,
              interval=interval,
              retrain=retrain)

def clf_train(data_path:str,
               out_path:str,
               clf_type="class_ens",
               interval=None,
               retrain=False):
    @utils.ParallelDirFun()
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        dir_proxy=base.get_dir_path(out_path=exp_path,
                                    clf_type=clf_type)
        clf_factory=clfs.get_clfs(clf_type)
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

def nn_train(data_path:str,
               out_path:str,
               clf_type="MLP",
               interval=None,
               retrain=False):
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        dir_proxy=base.get_dir_path(out_path=exp_path,
                                    clf_type=clf_type)
        clf_factory=clfs.get_clfs(clf_type)
        clf_factory.init(data)
        dir_proxy.make_dir(["history","models","results"])
        key = None if(retrain) else "models"
        path_dict=dir_proxy.path_dict(indexes=interval,
                                      key=key)
        print(path_dict['models'])
        if( len(path_dict['models'])==0):
            raise Exception("Model exists")
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


def train_only(data_path:str,
               out_path:str,
               clf_type="MLP",
               interval=None,
               retrain=False):
    if(not clf_type in base.NEURAL_CLFS):
        raise Exception(f"Unknown type {clf_type}")
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        dir_proxy=base.get_dir_path(out_path=exp_path,
                                    clf_type=clf_type)
        clf_factory=clfs.get_clfs(clf_type)
        clf_factory.init(data)
        dir_proxy.make_dir(["history","models"])
        key = None if(retrain) else "models"
        path_dict=dir_proxy.path_dict(indexes=interval,
                                      key=key)
        for i,split_path_i in tqdm(enumerate(path_dict['splits'])):
            clf_i=clf_factory()
            split_i=base.read_split(split_path_i)
            clf_i,history_i=split_i.fit_clf(data,clf_i)
            history_i= history_to_dict(history_i)
            with open(f"{path_dict['history'][i]}", 'w') as f:
                json.dump(history_i, f)
            clf_i.save(path_dict["models"][i])
        dir_proxy.save_info(clf_factory)
    helper(data_path,out_path)            

def pred_only(data_path:str,
              out_path:str,
              clf_type="MLP"):
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        info_path=f"{exp_path}/{clf_type}/info.js"
        clf_factory=clfs.read_factory(info_path)
        clf_factory.init(data)
        dir_proxy=base.get_dir_path(out_path=exp_path,
                                clf_type=clf_type)
        dir_proxy.make_dir(["results"])
        path_dict=dir_proxy.path_dict(indexes=interval,
                                      key=None)
        for i,model_path_i in tqdm(enumerate(path_dict['models'])):
            model_i=clf_factory.read(model_path_i)
            split_path_i=path_dict["splits"][i]
            split_i=base.read_split(split_path_i)
            result_i=split_i.pred(data,model_i)
            result_i.save(path_dict["results"][i])
    helper(data_path,out_path)            

def valid_clf(clf_type):
    is_neural= (clf_type in base.NEURAL_CLFS)
    is_other= (clf_type in base.OTHER_CLFS)
    if(not (is_neural or is_other)):
        raise Exception(f"Unknown clf {clf_type}")

def partial_pred( data_path:str,
                  out_path:str):
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        paths=base.get_ens_path(exp_path)
        for path_i in paths:
            print(path_i)
            factor_i=clfs.read_factory(f"{path_i}/info.js")
            model_paths=utils.top_files(f"{path_i}/models")
            for j,model_path_j in tqdm(enumerate(model_paths)):
                model_i=factor_i.read(model_path_j)
                
    helper(data_path,out_path)            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="uci_exp/data")
    parser.add_argument("--out_path", type=str, default="uci_exp/exp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=10)
#    parser.add_argument('--retrain', action='store_true')
    parser.add_argument("--clf_type", type=str, default="BINARY-CS-TREE-ENS")
    args = parser.parse_args()
    print(args)
#   valid_clf(args.clf_type)
#    train(data_path=args.data,
#          out_path=args.out_path,
#          start=args.start,
#          step=args.step,
#          clf_type=args.clf_type,
#          retrain=False)
    partial_pred( data_path=args.data,
                  out_path=args.out_path)