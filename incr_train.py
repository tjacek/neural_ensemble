import numpy as np
#import json
from tqdm import tqdm
import base,clfs,dataset,plot,utils

def incr_train(in_path,
               exp_path,
               hyper_path,
               n=2,
               n_splits=30,
               selected=None):
    selector=get_selector(selected,pos=True)
    build_clf=get_factory(hyper_path)
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        name=in_path.split("/")[-1]
        if(selector(name)):
            return
        print(name)
        data=dataset.read_csv(in_path)
        clf_factory=build_clf(name,n)
        model_path=prepare_dirs(exp_path)
        clf_factory.init(data)
        gen=splits_gen(exp_path,n_splits)
        for i,split_i in tqdm(gen):
            clf_i=clf_factory()
            clf_i,history_i=split_i.fit_clf(data,clf_i)
            save_incr(clf_i,f"{model_path}/{i}")
            print(split_i)	
    helper(in_path,exp_path)

def get_selector(selected,pos=True):
    if(selected is None):
        return lambda name:False
    selected=set(selected)
    if(pos):
        return lambda name: name in selected
    else:
        return lambda name: not name in selected

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

def get_factory(hyper_path):
    hyper_dict=utils.read_json(hyper_path)
    def build_clf(name,n):
        info_dict=hyper_dict[name]
        return clfs.get_clfs( clf_type=f'TREE-ENS({n})',
                              hyper_params=info_dict["hyper"],
                              feature_params=info_dict["feature_params"])
    return build_clf

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


def model_iter(clf_factory,exp_path,n_splits=30):
    model_path=prepare_dirs(exp_path)
    if(not utils.top_files(model_path)):
        raise Exception(f"No models at {model_path}")
    for i,split_i in splits_gen(exp_path,n_splits):
        clf_i=clf_factory.read(f"{model_path}/{i}")
        yield clf_i,split_i

def incr_pred( in_path,
               exp_path,
               hyper_path):
    build_clf=get_factory(hyper_path)
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        model_path=prepare_dirs(exp_path)
        if(not utils.top_files(model_path)):
            return None
        name=in_path.split("/")[-1]
        print(name)
        clf_factory=build_clf(name,n=2)
        clf_factory.init(data)
        gen=model_iter(clf_factory,exp_path)
        all_results=[]
        for clf_i,split_i in tqdm(gen):
            result_i=split_i.pred(data,clf_i)
            all_results.append(result_i)
        result_group=dataset.ResultGroup(all_results)
        result_group.save(f"{exp_path}/TREE-ENS/results")
        return np.mean(result_group.get_acc())
    output_dict=helper(in_path,exp_path)
    print(output_dict)
    
def incr_partial( in_path,
                  exp_path,
                  hyper_path):
    build_clf=get_factory(hyper_path)
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        data=dataset.read_csv(in_path)
        name=in_path.split("/")[-1]
        print(name)
        clf_factory=build_clf(name,n=2)
        clf_factory.init(data)
        all_partials=[]
        gen=model_iter(clf_factory,exp_path)
        for clf_i,split_i in tqdm(gen):
            partial_i=split_i.pred_partial(data,clf_i)
            all_partials.append(partial_i)
        part_group=dataset.PartialGroup(all_partials)
        print(part_group.indv_acc())
        return part_group
    output_dict=helper(in_path,exp_path)
    rf_dict={"gesture":0.6797,"first-order":0.6299,
             "wine-quality-white":0.6954,
             "wine-quality-red":0.7155}
    for name_i,partial_i in output_dict.items():
        print(name_i)
        partial_series=partial_i.subset_series(step=5)
        print(partial_series.steps)
        print(partial_series.means)
        print(partial_series.stds)
        plot.error_plot(partial_series.steps,
                        partial_series.means,
                        partial_series.stds,
                        name=name_i,
                        vertical=rf_dict[name_i],
                        xlabel="n_clfs",
                        ylabel="accuracy")
if __name__ == '__main__':
    in_path="incr_exp/multi/_data"
    hyper_path="incr_exp/multi/hyper.js"
    incr_train(in_path,
               "incr_exp/multi/exp",
               hyper_path,
               3)
    incr_pred(in_path,"incr_exp/multi/exp",hyper_path)
#    incr_partial(in_path,"bad_exp/exp",hyper_path)