import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import pred,plot,utils


def metric_plot(conf_dict):
    metric,text=conf_dict["metric"],conf_dict["text"]
    x_clf,y_clf=conf_dict["x_clf"],conf_dict["y_clf"]
    result_dict=pred.unify_results(conf_dict["exp_path"])
    x_dict=result_dict.get_mean_metric(x_clf,metric=metric)
    y_dict=result_dict.get_mean_metric(y_clf,metric=metric)
    if("names" in conf_dict):
        text=conf_dict["names"]
    plot.dict_plot( x_dict,
                    y_dict,
                    xlabel=f"{x_clf}({metric})",
                    ylabel=f"{y_clf}({metric})",
                    text=text)

def count_feats(conf_dict):
    @utils.DirFun(out_arg=None)
    def helper(in_path):
        model_path=f"{in_path}/TREE-MLP/models"
        n_feats=[ feat_i.shape[0] 
                  for feat_i in iter_feats(model_path)] 
        return np.mean(n_feats),np.std(n_feats)
    for exp_i in conf_dict['exp_path']:
        output=helper(exp_i)
        for name_i,value_i in output.items():
            print(f"{name_i},{value_i}")

def feat_hist(in_path):
    @utils.DirFun(out_arg=None)
    def helper(in_path):
        feat_dict=defaultdict(lambda:0)
        model_path=f"{in_path}/TREE-MLP/models"
        for feat_i in iter_feats(model_path):
            for feat_j  in feat_i:
                feat_dict[feat_j]+=1
        keys=list(feat_dict.keys())
        keys.sort()
        y=[feat_dict[key_i] for key_i in keys]
        plt.bar(keys,y)
        plt.title(in_path.split("/")[-1])
        plt.show()

    helper(in_path)    

def iter_feats(model_path):
    for model_i in utils.top_files(model_path):
        feat_path_i=f"{model_i}/tree/feats.npy"
        feat_i=np.load(feat_path_i)
        yield feat_i

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="sum.json")
    args = parser.parse_args()
    conf_dict=utils.read_json(args.conf)
#    metric_plot(conf_dict)
    feat_hist("uci_exp/exp")