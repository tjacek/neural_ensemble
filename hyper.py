import numpy as np
from sklearn import tree
import base

class TreeFeatures(object):
    def __init__(self,features,thresholds):
        self.features=features
        self.thresholds=thresholds

#    def 

def make_tree_feats(tree):
    tree_repr=tree.tree_.__getstate__()['nodes']

def gradient_tree():
    return tree.DecisionTreeClassifier(max_depth=3,
                                       class_weight="balanced")

def eval_features(in_path):
    data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
    acc=[]
    for clf_i,result_i in data_split.eval(gradient_tree):
#        result_i,_=split_i.eval(data_split.data,
#                      gradient_tree())
        acc.append(result_i.get_acc())
    print(np.mean(acc))

eval_features("bad_exp/data/wine-quality-red")