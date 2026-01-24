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

def indv_acc(in_path):
    paths=utils.get_paths(in_path)
    @utils.DirFun("in_path",[])
    def helper(in_path):
        path_i=f"{in_path}/TreeEnsTabPF/partials"
        partials=dataset.PartialGroup.read(path_i)
        acc=partials.indv_acc()
        print(np.mean(acc,axis=0))
    helper(in_path)

def incr_train( in_path,
                out_path,
                hyper_path,
                n_clfs=2):
    paths=utils.get_paths(in_path)
    hyper_dict=hyper.BestHyper.read(hyper_path)
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
        hyper_i["n_clfs"]=n_clfs
        clf_factory=tree_clf.TreeEnsFactory(hyper_i)
        all_partials=[]
        for i,split_i in tqdm(enumerate(splits)):
            clf_i = clf_factory()
            clf_i,_=split_i.fit_clf(data,clf_i)
            partial_i=split_i.predict_partial(data,clf_i)
            all_partials.append(partial_i)
        result=dataset.PartialGroup(all_partials)
        clf_path=f"{out_path}/{str(clf_factory)}"
        utils.make_dir(clf_path)
        partial_path=f"{clf_path}/partials"
        result.save(partial_path)
        full_partials=dataset.PartialGroup.read(partial_path)
        full_result=full_partials.to_result()
        full_result.save(f"{clf_path}/results")
        print(full_partials.n_clfs())
        print(np.mean(full_result.get_acc()))
    helper(in_path,out_path)

def train_exps(dirs,n_clf=2):
    for dir_i in dirs:
        incr_train( in_path=f"{dir_i}/data",
                    out_path=f"{dir_i}/exp",
                    hyper_path=f"{dir_i}/hyper.csv",
                    n_clfs=n_clfs):

def clf_exp(dirs):
    clfs=["TabPF"]
    for clf_i in clf:
        for dir_j in dirs:
            train( in_path=f"{dir_j}/data",
                   out_path=f"{dir_j}/exp",
                   hyper_path=f"{dir_j}/hyper.csv",
                   factory_type=clf_i)
paths=["test/A","test/B"]
#incr_train(in_path=paths,
#	       out_path="test_exp",
#	       hyper_path="hyper.csv",
#           n_clfs=2)