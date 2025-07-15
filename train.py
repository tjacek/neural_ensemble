import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
import os.path
import argparse,json
import base,clfs,dataset,utils

def train(data_path:str,
              out_path:str,
              clf_type="class_ens",
              start=0,
              step=10):
    if(clf_type in base.NEURAL_CLFS):
        train_fun=nn_train
    else:
        train_fun=clf_train
    train_fun(data_path=data_path,
              out_path=out_path,
              clf_type=clf_type,
              start=start,
              step=step)

def clf_train(data_path:str,
               out_path:str,
               clf_type="class_ens",
               start=0,
               step=10):
    interval=base.Interval(start,step)
    @utils.ParallelDirFun()#("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        dir_proxy=base.get_dir_path(out_path=exp_path,
                                    clf_type=clf_type)
        clf_factory=clfs.get_clfs(clf_type)
        path_dict=dir_proxy.path_dict(indexes=interval,
                                      key="results")
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
               clf_type="class_ens",
               start=0,
               step=10):
    interval=base.Interval(start,step)
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        dir_proxy=base.get_dir_path(out_path=exp_path,
                                    clf_type=clf_type)
        clf_factory=clfs.get_clfs(clf_type)
        clf_factory.init(data)
        dir_proxy.make_dir(["history","models","results"])
        path_dict=dir_proxy.path_dict(indexes=interval,
                                         key="models")
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="bad_exp/data")
    parser.add_argument("--out_path", type=str, default="bad_exp/exp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--clf_type", type=str, default="RF")
    args = parser.parse_args()
    print(args)
    train(data_path=args.data,
          out_path=args.out_path,
          start=args.start,
          step=args.step,
          clf_type=args.clf_type)