from tqdm import tqdm
import base,clfs,dataset,utils

def incr_train(in_path,split_path="split",n=5):
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
      
        clf_factory=clfs.get_clfs(clf_type=f'TREE-ENS({n})',
                    	          hyper_params=None,
                                  feature_params=None)
        
        model_path=prepare_dirs(exp_path)
        clf_factory.init(data)
        for i,split_i in tqdm(splits_gen(exp_path)):
            clf_i=clf_factory()
            clf_i,history_i=split_i.fit_clf(data,clf_i)
            clf_i.save(f"{model_path}/{i}")
            print(split_i)	
    helper(in_path,"bad_exp/exp")

def prepare_dirs(exp_path):
    utils.make_dir(f"{exp_path}/TREE-ENS")
    model_path=f"{exp_path}/TREE-ENS/models"
    utils.make_dir(model_path)
    return model_path

def splits_gen(exp_path,
               n_splits=10,
               start=0):
    split_path=f"{exp_path}/splits"
    end=start+n_splits
    paths=utils.top_files(split_path)[start:end]
    for i,split_path_i in enumerate(paths):
        split_i=base.read_split(split_path_i)
        yield start+i,split_i

if __name__ == '__main__':
    in_path="bad_exp/data"
    incr_train(in_path)
