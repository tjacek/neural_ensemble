import numpy as np
from tqdm import tqdm
import os.path
import base,dataset,hyper,tree_clf,utils

class HyperIter(object):
    def __init__(self,hyper_space=None):
        if(hyper_space is None):
            hyper_space=hyper.HyperparamSpace()
        self.hyper_space=hyper_space

    def __call__(self,out_path):
        for hyper_i in hyper_space():
            hyper_id=params_id(hyper_i)
            out_i=f"{out_path}/{hyper_id}"
            if(not os.path.isfile(out_i)):
                yield hyper_i,out_i

def make_archive( in_path,
	              out_path,
	              n_iters=3,
	              n_clfs=2):
    utils.make_dir(out_path)
    @utils.DirFun("in_path",["out_path"])
    def helper(in_path,out_path):
        utils.make_dir(out_path)
        data=dataset.read_csv(in_path)
        splits=base.get_splits(data,10,n_iters)
        base.save_splits(f"{out_path}/splits",splits)
        hyper_space=hyper.HyperparamSpace()
        for hyper_i in hyper_space():
            hyper_id=params_id(hyper_i)
            hyper_i["n_clfs"]=n_clfs
            eval_hyper(data,
            	       splits,
            	       hyper_i,
            	       f"{out_path}/{hyper_id}")
            print(hyper_id)
    helper(in_path,out_path)

def eval_hyper( data,
	            splits,
	            hyper,
	            out_path):
    utils.make_dir(out_path)
    clf_factory=tree_clf.TreeEnsFactory(hyper)
    all_partials=[]
    for i,split_i in tqdm(enumerate(splits)):
        clf_i = clf_factory()
        clf_i,_=split_i.fit_clf(data,clf_i)
        partial_i=split_i.predict_partial(data,clf_i)
        all_partials.append(partial_i)
    result=dataset.PartialGroup(all_partials)
    result.save(out_path)

def params_id(hyper_dict):
	keys=list(hyper_dict.keys())
	keys.sort()
	values=[str(hyper_dict[key_i]) 
	        for key_i in keys]
	return "_".join(values)

def result_iter(in_path):
    for path_i in utils.top_files(in_path):
        dir_id=path_i.split("/")[-1]
        if(dir_id=="splits"):
            continue
        yield path_i,dir_id

def show_archive(in_path):
    @utils.DirFun("in_path")
    def helper(in_path):
        data_id=in_path.split("/")[-1]
        print(data_id)
        for  path_i,dir_id in result_iter(in_path):
            parital=dataset.PartialGroup.read(path_i)	
            result=parital.to_result()
            acc=result.get_acc()
            balance=result.get_balanced()
            metrics=f"{np.mean(acc):.4f},{np.mean(balance):.4f}"
            line=f"{data_id},{dir_id},{metrics}"
            print(line)
    helper(in_path)

def hyper_var(in_path,n_splits=10):
    @utils.DirFun("in_path")
    def helper(in_path):
        acc_dict={}
        for path_i,dir_id in result_iter(in_path):
#            print(dir_id)
            parital=dataset.PartialGroup.read(path_i)
            sub_results=parital.split_results(n_splits)
            for res_i in sub_results:
                acc_i=[ res_j.get_mean("acc")
                         for res_j in res_i]
                acc_dict[dir_id]=acc_i
        return acc_dict
    output=helper(in_path)
    for name_i,dict_i in output.items():
        print(name_i)
        params,acc_i=[],[]
        for params_j,acc_j in dict_i.items():
            params.append(params_j)
            acc_i.append(acc_j)
        acc_i=np.array(acc_i)
        for acc_t in acc_i.T:
            k=np.argmax(acc_t)
            print(params[k])
            print(acc_i[k])

paths=["test/A","test/B"]
#make_archive(paths,"archive",n_clfs=1)
#show_archive("archive")
hyper_var("archive")