from tqdm import tqdm
import base,clfs,dataset,utils

def incr_train(in_path,split_path="split",n=5):
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        split_path=f"{exp_path}/splits"
        paths=utils.top_files(split_path)[:10]
        clf_factory=clfs.get_clfs(clf_type=f'TREE-ENS({n})',
                    	          hyper_params=None,
                                  feature_params=None)
        clf_factory.init(data)
        for i,split_path_i in tqdm(enumerate(paths)):
            split_i=base.read_split(split_path_i)
            clf_i=clf_factory()
            clf_i,history_i=split_i.fit_clf(data,clf_i)
            print(split_i)	
#	    dir_proxy=base.get_dir_path(out_path=exp_path,
#                                    clf_type="TREE-ENS")
#	    print(dir_proxy)
    helper(in_path,"bad_exp/exp")

#	for i,split_path_i in tqdm(enumerate(split_path)):
#            clf_i=clf_factory()
#            split_i=base.read_split(split_path_i)
#            clf_i,history_i=split_i.fit_clf(data,clf_i)

if __name__ == '__main__':
    in_path="bad_exp/data"
    incr_train(in_path)
