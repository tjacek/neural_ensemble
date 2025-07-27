import numpy as np
from sklearn import tree
import base,dataset

class TreeFeatClf(object):
    def __init__(self,extract_feats,
                      clf_type,
                      concat=False):
        self.extract_feats=extract_feats
        self.clf_type=clf_type
        self.concat=concat
        self.clf=None

    def fit(self,X,y):
        self.extract_feats.fit(X,y)
        self.clf=base.get_clf(self.clf_type)
        X=self.extract_feats(X=X,
                             concat=self.concat)
        self.clf.fit(X,y)

    def predict(self,X):
        X=self.extract_feats(X=X,
                          concat=self.concat)
        return self.clf.predict(X)

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

class FeatureExtractor(object):
    def __init__(self,tree_factory=None):
        if(tree_factory is None):
            tree_factory=gradient_tree
        self.tree_factory=tree_factory
        self.tree_feats=None

    def fit(self,X,y):
        tree=self.tree_factory()
        tree.fit(X,y)
        self.tree_feats=make_tree_feats(tree)

    def __call__(self,X,concat=True):
        return self.tree_feats(X,concat)
        
class CSExtractor(object):
    def __init__(self,tree_factory=None):
        if(tree_factory is None):
            tree_factory=gradient_tree
        self.tree_factory=tree_factory
        self.indiv__feats=[]

    def fit(self,X,y):
        data= dataset.Dataset(X,y)
        for i in range(data.n_cats()):
            data_i=data.binarize(i)
            tree_i=self.tree_factory()
            tree_i.fit(X=data_i.X,
                       y=data_i.y)
            feats_i=make_tree_feats(tree_i)
            self.indiv__feats.append(feats_i)

    def __call__(self,X,concat=True):
        feats=[feats_i(X,concat=False) 
                  for feats_i in self.indiv__feats]
        if(concat):
            feats.append(X)
        feats=np.concatenate(feats,axis=1)
        return feats

def gradient_tree():
    return tree.DecisionTreeClassifier(max_depth=3,
                                       class_weight="balanced")

def eval_features(in_path):
    def helper():
        return TreeFeatClf(extract_feats=CSExtractor(),
                           clf_type="LR",
                           concat=True)
    clf_dict={
              "TREE":gradient_tree,
              "TREE-SVM":helper,
              "SVM":"SVM"}
    compare_clf(in_path,clf_dict)

def compare_clf(in_path,clf_dict):
    data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
    for name_i,clf_factory_i in clf_dict.items():
        if(type(clf_factory_i)==str):
            clf_factory_i=base.ClasicalClfFactory(clf_factory_i)
        acc_i=[]
        for clf_j,result_j in data_split.eval(clf_factory_i):
            acc_i.append(result_j.get_acc())
        print(f"{name_i}:{np.mean(acc_i):.4f}")

eval_features("bad_exp/data/wine-quality-red")