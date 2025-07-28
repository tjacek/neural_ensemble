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

class TabFeatures(object):
    def __call__(self,X,concat=True):
        new_feats=[self.compute_feats(x_i) for x_i in X]
        new_feats=np.array(new_feats)
        if(concat):
            return np.concatenate([X,new_feats],axis=1)
        return new_feats 

    def compute_feats(self,x_i):
        raise NotImplementedError()

class TreeFeatures(TabFeatures):
    def __init__(self,features,thresholds):
        self.features=features
        self.thresholds=thresholds

    def n_feats(self):
        return len(self.features)

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

class ThresholdFeats(TabFeatures):
    def __init__(self,thres_dict):
        self.thres_dict=thres_dict
#    def __call__(self,X,concat=True):
    
    def compute_feats(self,x):
        new_feats=[]
        for feat_i,x_i in enumerate(x):
            if(feat_i in self.thres_dict):
                thres_i=self.thres_dict[feat_i]
                value_i=None
                for j,thres_j in enumerate(thres_i):
                    if(x_i<thres_j):
                        value_i=j
                        break
                if(value_i is None):
                    value_i=len(thres_i)
                new_feats.append(value_i)
        return new_feats

    def propor(self):
        keys=list(self.thres_dict.keys())
        keys.sort()
        for key_i in keys:
            thres_i=self.thres_dict[key_i]
            thres_i-=thres_i[0]
            thres_i/=thres_i[-1]
            thres_i=np.round(thres_i,4)
            print(thres_i)

def make_thres_feats(tree):
    raw_tree=tree.tree_.__getstate__()['nodes']
    thres_dict=defaultdict(lambda:[])
    for node_i in raw_tree:
        feat_i=node_i[2]
        if(feat_i>=0):
            thres_dict[feat_i].append(node_i[3])
    new_dict={}
    for feat_i,thres_i in thres_dict.items():
        thres_i.sort()
        thres_i=np.array(thres_i)
        new_dict[feat_i]=thres_i
    return ThresholdFeats(new_dict)

def thre_stats(X):
    for feat_i in X.T:
#        print(x_i.shape)
        print(np.unique(feat_i).shape)

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    clf=get_tree("random")()
    clf.fit(data.X,data.y)
    thres_feat=make_thres_feats(clf)
    new_X=thres_feat(data.X,concat=False)
    thre_stats(new_X)
