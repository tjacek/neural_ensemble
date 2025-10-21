import dataset,utils

def incr_train(in_path,n):
	data=dataset.read_csv(in_path)
	clf_factory=clfs.get_clfs(clf_type=f'TREE-ENS({n})',
        	                  hyper_params=hyper_params,
                              feature_params=feature_params)