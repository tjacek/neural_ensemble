import numpy as np
from collections import defaultdict
from sklearn import tree

def get_tree(tree_type):
    if(tree_type=="random"):
        return RandomTree()
    return GradientTree()

class GradientTree(object):
    def __call__(self):
        return tree.DecisionTreeClassifier(max_depth=3,
                                       class_weight="balanced")

    def __str__(self):
        return "GradientTree"

class RandomTree(object):
    def __call__(self):
        return tree.DecisionTreeClassifier(max_features='sqrt')#,
#                                           class_weight="balanced")

    def __str__(self):
        return "RandomTree"

class TreeFeatures(object):
    def __init__(self,features,thresholds):
        self.features=features
        self.thresholds=thresholds

    def n_feats(self):
        return len(self.features)

    def __call__(self,X,concat=True):
        new_feats=[self.compute_feats(x_i) for x_i in X]
        new_feats=np.array(new_feats)
        if(concat):
            return np.concatenate([X,new_feats],axis=1)
        return new_feats

    def compute_feats(self,x_i):
        new_feats=[]
        for i,feat_i in enumerate(self.features):
            value_i=x_i[feat_i]
            thres_i=self.thresholds[i]
            new_feats.append(int(value_i<thres_i) )
        return np.array(new_feats)

def make_tree_feats(tree):
    raw_tree=tree.tree_.__getstate__()['nodes']
    feats,thres=[],[]
    for node_i in raw_tree:
        feat_i=node_i[2]
        if(feat_i>=0):
            feats.append(feat_i)
            thres.append(node_i[3])
    return TreeFeatures(feats,thres)

#class ThresholdFeats(object):

def make_thres_feats(tree):
    raw_tree=tree.tree_.__getstate__()['nodes']
    thres_dict=defaultdict(lambda:[])
    for node_i in raw_tree:
        feat_i=node_i[2]
        if(feat_i>=0):
            thres_dict[feat_i].append(node_i[3])
    for feat_i,thres_i in thres_dict.items():
        thres_i.sort()
        thres_i=np.array(thres_i)
        print(feat_i)
        thres_i-=thres_i[0]
        print( thres_i/thres_i[-1])

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    clf=get_tree("random")()
    clf.fit(data.X,data.y)
    make_thres_feats(clf)