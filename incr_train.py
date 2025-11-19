import numpy as np
#import json
from tqdm import tqdm
import base,clfs,dataset,plot,utils

def incr_train(in_path,
               exp_path,
               hyper_path,
               n=2,
               n_splits=30,
               selected=None,
               pos=False):
    selector=get_selector(selected,pos=pos)
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
               hyper_path,
               selected):
    selector=get_selector(selected,pos=False)

    build_clf=get_factory(hyper_path)
    @utils.DirFun("in_path","exp_path")
    def helper(in_path,exp_path):
        name=in_path.split("/")[-1]
        if(selector(name)):
            return
        print(name)
        data=dataset.read_csv(in_path)
        model_path=prepare_dirs(exp_path)
        if(not utils.top_files(model_path)):
            return None
        clf_factory=build_clf(name,n=2)
        clf_factory.init(data)
        gen=model_iter(clf_factory,exp_path)
        all_partials,all_results=[],[]
        for clf_i,split_i in tqdm(gen):
            partial_i=split_i.pred_partial(data,clf_i)
            all_partials.append(partial_i)
            result_i=partial_i.to_result()
            all_results.append(result_i)
        partial_group=dataset.PartialGroup(all_partials)
        partial_group.save(f"{exp_path}/TREE-ENS/partials")
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
        path_i=f"{exp_path}/TREE-ENS/partials"
        part_i=dataset.PartialGroup.read(path_i)
        print(part_i.indv_acc())
        return part_i
    output_dict=helper(in_path,exp_path)
    rf_dict=RF_acc(exp_path)
    print(rf_dict)
    for name_i,partial_i in output_dict.items():
        partial_series=partial_i.subset_series(step=2)
        plot.error_plot(partial_series.steps,
                        partial_series.means,
                        partial_series.stds,
                        name=name_i,
                        vertical=rf_dict[name_i],
                        xlabel="n_clfs",
                        ylabel="accuracy")

def RF_acc(exp_path):
    @utils.DirFun("exp_path",None)
    def helper(exp_path):
        result_path=f"{exp_path}/RF/results"
        result=dataset.ResultGroup.read(result_path)
        return np.mean(result.get_acc())
    return helper(exp_path)

def clf_count(exp_path):
    for path_i in utils.top_files(exp_path):
        name_i=path_i.split("/")[-1]
        tree_path=f"{path_i}/TREE-ENS/models"
        count=[ len(utils.top_files(tree_i))
                  for tree_i in utils.top_files(tree_path)]
        print(name_i)
        print(count)

if __name__ == '__main__':
    in_path="incr_exp/uci/data"
    exp_path="incr_exp/multi/exp"
    hyper_path="incr_exp/uci/hyper.js"
    selected=[ 'wine-quality-white',
               'first-order']
#    incr_train(in_path,
#               exp_path,
#               hyper_path,
#               2,
#               selected=selected)
#    incr_pred(in_path,
#              exp_path,
#              hyper_path,
#              selected=None)
#    clf_count(exp_path)
    incr_partial(in_path,"incr_exp/uci/exp",hyper_path)