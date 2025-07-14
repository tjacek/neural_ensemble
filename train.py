import utils
utils.silence_warnings()
import numpy as np
from tqdm import tqdm
import os.path
import argparse,json
import base,clfs,dataset,utils

#    def get_paths(self,dir_path:str):
#        indexs=[self.start+j for j in range(self.step)]
#        paths=[ (index,f"{dir_path}/{index}.keras") 
#               for index in indexs]
#        if(not os.path.isdir(model_path)):
#                return paths
#        paths=[ (i,model_i)
#                for i,model_i in paths
#                    if(not (os.path.isfile(model_i) or
#                         os.path.isdir(model_i)))]
#        return paths

def train(data_path:str,
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
        splits=dir_proxy.get_splits(interval)
        dir_proxy.make_dir("results")
        result_paths=dir_proxy.get_paths(interval,
                            key="results")
        for i,split_i in tqdm(enumerate(splits)):
            clf_i=clf_factory()
            split_i.fit_clf(data,clf_i)
            result_i=clf_i.eval(data,split_i)
            result_i.save(result_paths[i])
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
#        splits=dir_proxy.get_splits(indexes)
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

#def pred_clf(data_path:str,
#                 exp_path:str,
#                 clf_type="RF"):
#    @utils.MultiDirFun()
#    def helper(in_path,exp_path):
#        data=dataset.read_csv(in_path)
#        path_dir=base.get_paths(out_path=exp_path,
#                                 ens_type=clf_type,
#                                 dirs=['results','info.js'])
#        utils.make_dir(path_dir["ens"])
#        clf_factory=ens.get_ens(clf_type)
#        split_path=utils.top_files(path_dir['splits'])
#        utils.make_dir(path_dir["results"])
#        for i,split_path_i in tqdm(enumerate(split_path)):
#            split_i=base.read_split(split_path_i)
#            clf_i=clf_factory()
#            split_i.fit_clf(data,clf_i)
#            result_i=clf_i.eval(data,split_i)
#            result_i.save(f"{path_dir['results']}/{i}.npz")
#        utils.save_json(value=clf_factory.get_info(),
#                        out_path=path_dir['info.js'])
#    helper(data_path,exp_path)


#def nn_train(data_path:str,
#               out_path:str,
#               ens_type="class_ens",
#               start=0,
#               step=10):
#    path_dict=base.get_paths(out_path=out_path,
#                             ens_type=ens_type,
#                             dirs=['models','history','info.js'])
#    utils.make_dir(path_dict['ens'])    
#    model_paths=get_model_paths(path_dict['models'],start,step)
#    print(model_paths)
#    if(len(model_paths)==0):
#        raise Exception("Models exist")
#    utils.make_dir(path_dict['models'])
#    utils.make_dir(path_dict['history'])
#    data=dataset.read_csv(data_path)
#    clf_factory=clfs.get_clfs(ens_type)
#    clf_factory.init(data)
#    for index,model_path in tqdm(model_paths):
#        raw_split=np.load(f"{path_dict['splits']}/{index}.npz")
#        split_j=base.UnaggrSplit.Split(train_index=raw_split["arr_0"],
#                                       test_index=raw_split["arr_1"])
#        clf_j=clf_factory()
#        clf_j,history_j=split_j.fit_clf(data,clf_j)  
#        hist_dict_j=utils.history_to_dict(history_j)
#        clf_j.save(model_path)
#        with open(f"{path_dict['history']}/{index}", 'w') as f:
#            json.dump(hist_dict_j, f)
#    with open(path_dict['info.js'], 'w') as f:
#        json.dump(clf_factory.get_info(),f)

#def get_model_paths(model_path,start,step):
#    indexs=[start+j for j in range(step)]
#    paths=[ (index,f"{model_path}/{index}.keras") 
#               for index in indexs]
#    if(not os.path.isdir(model_path)):
#        return paths
#    paths=[ (i,model_i)
#              for i,model_i in paths
#                  if(not (os.path.isfile(model_i) or
#                         os.path.isdir(model_i)))]
#    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="bad_exp/data")
    parser.add_argument("--out_path", type=str, default="bad_exp/exp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--clf_type", type=str, default="MLP")
    args = parser.parse_args()
    print(args)
    nn_train(data_path=args.data,
          out_path=args.out_path,
          start=args.start,
          step=args.step,
          clf_type=args.clf_type)