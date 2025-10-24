import numpy as np
from tqdm import tqdm
import base,clfs,dataset,utils

def incr_train(in_path,exp_path,n=5):
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
      
        clf_factory=clfs.get_clfs(clf_type=f'TREE-ENS({n})',
                    	          hyper_params=None,
                                  feature_params=None)
        
        model_path,_=prepare_dirs(exp_path)
        clf_factory.init(data)
        for i,split_i in tqdm(splits_gen(exp_path)):
            clf_i=clf_factory()
            clf_i,history_i=split_i.fit_clf(data,clf_i)
#            clf_i.save(f"{model_path}/{i}")
            save_incr(clf_i,f"{model_path}/{i}")
            print(split_i)	
    helper(in_path,"bad_exp/exp")


def save_incr(clf,out_path):
    utils.make_dir(out_path)
    offset=len(utils.top_files(out_path))
    for i,clf_i in enumerate(clf.all_clfs):
        k=offset+i
        out_k=f"{out_path}/{k}"
        utils.make_dir(out_k)
        extr_i=clf.all_extract[i]
        extr_i.save(f"{out_k}/tree")
        clf_i.save(f"{out_k}/nn.keras")

#def get_factory(n):


def prepare_dirs(exp_path):
    utils.make_dir(f"{exp_path}/TREE-ENS")
    model_path=f"{exp_path}/TREE-ENS/models"
    utils.make_dir(model_path)
    result_path=f"{exp_path}/TREE-ENS/results"
    utils.make_dir(model_path)
    return model_path,result_path

def splits_gen(exp_path,
               n_splits=10,
               start=0):
    split_path=f"{exp_path}/splits"
    end=start+n_splits
    paths=utils.top_files(split_path)[start:end]
    for i,split_path_i in enumerate(paths):
        split_i=base.read_split(split_path_i)
        yield start+i,split_i

def incr_pred(n_path,exp_path):
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        model_path,result_path=prepare_dirs(exp_path)
        clf_factory=clfs.get_clfs(clf_type=f'TREE-ENS',
                                  hyper_params=None,
                                  feature_params=None)
        clf_factory.init(data)
        all_results=[]
        for i,split_i in splits_gen(exp_path):
            clf_i=clf_factory.read(f"{model_path}/{i}")
            print(clf_i)
            result_i=split_i.pred(data,clf_i)
            all_results.append(result_i)
            print(result_i.get_acc())
        all_results=dataset.ResultGroup(all_results)
        return np.mean(all_results.get_acc())
    output_dict=helper(in_path,exp_path)
    print(output_dict)

if __name__ == '__main__':
    in_path="bad_exp/data"
    incr_train(in_path,"bad_exp/exp",4)
    incr_pred(in_path,"bad_exp/exp")