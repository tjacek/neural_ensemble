import numpy as np
from tqdm import tqdm
import dataset
import base,hyper,utils,tree_clf

def train( in_path,
           out_path,
           hyper_path,
           factory_type="TabPF"):
    paths=utils.get_paths(in_path)
    hyper_dict=hyper.read_hyper(hyper_path)
    @utils.DirFun("in_path",["out_path"])
    def helper(in_path,out_path):
        utils.make_dir(out_path)
        base.make_split_dir(in_path,out_path)
        data_id=in_path.split("/")[-1]
        hyper_i=hyper_dict[data_id]
        splits=base.read_split_dir(f"{out_path}/splits")
        clf_factory=tree_clf.get_factory(factory_type)()#(hyper_i)
        split_iter=enumerate(splits)
        result=clf_factory.get_results(in_path,
                                       split_iter,
                                       hyper_i)
        print(hyper_i)
        print(np.mean(result.get_acc()))
        pred_path=f"{out_path}/{str(clf_factory)}"
#        utils.make_dir(pred_path)
#        result.save(f"{pred_path}/results")
    helper(in_path,out_path)


def incr_train( in_path,
                out_path,
                hyper_path,
                n_clfs=2):
    paths=utils.get_paths(in_path)
    hyper_dict=hyper.read_hyper(hyper_path)     
    @utils.DirFun("in_path",["out_path"])
    def helper(in_path,out_path):
        print(in_path)
        print(out_path)
        utils.make_dir(out_path)
        base.make_split_dir(in_path,out_path)
        data=dataset.read_csv(in_path)
        data_id=in_path.split("/")[-1]
        hyper_i=hyper_dict[data_id]
        splits=base.read_split_dir(f"{out_path}/splits")
        hyper_i["n_clfs"]=2
        clf_factory=tree_clf.TreeEnsFactory(hyper_i)
        for i,split_i in tqdm(enumerate(splits)):
            clf_i = clf_factory()
            clf_i,_=split_i.fit_clf(data,clf_i)
            result_i=split_i.predict_partial(data,clf_i)
            raise Exception(result_i)
            print(split_i)
    helper(in_path,out_path)

paths=["test/A","test/B"]
incr_train(in_path=paths,
	       out_path="test_exp",
	       hyper_path="hyper_goodII.csv")
