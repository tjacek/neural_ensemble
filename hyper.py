import numpy as np
from sklearn.decomposition import PCA
import base,dataset,tree_feats,utils
utils.silence_warnings()

class TreeFeatFactory(object):
    def __init__(self,arg_dict):
        self.arg_dict=arg_dict

    def __call__(self):
        return TreeFeatClf(**self.arg_dict)

class TreeFeatClf(object):
    def __init__(self,
                 tree_type,
                 extract_feats,
                 clf_type,
                 concat=False):
        tree_factory=tree_feats.get_tree(tree_type)
        extract_feats=get_extractor(extract_feats)
        self.extract_feats=extract_feats(tree_factory)
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

def get_extractor(extr_feats):
    if(extr_feats=="CS"):
        return CSExtractor
    if(extr_feats=="PCA"):
        return PCAExtractor
    if(extr_feats=="thres"):
        return ThresExtractor
    return FeatureExtractor

class FeatureExtractor(object):
    def __init__(self,tree_factory=None):
        if(tree_factory is None):
            tree_factory=gradient_tree
        self.tree_factory=tree_factory
        self.tree_feats=None

    def fit(self,X,y):
        tree=self.tree_factory()
        tree.fit(X,y)
        self.tree_feats=tree_feats.make_tree_feats(tree)

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

class PCAExtractor(object):
    def __init__(self,tree_factory=None):
        if(tree_factory is None):
            tree_factory=gradient_tree
        self.tree_factory=tree_factory
        self.pca_feats=None

    def fit(self,X,y):
        tree=self.tree_factory()
        tree.fit(X,y)
        self.tree_feats=tree_feats.make_tree_feats(tree)
        n_feats=2*X.shape[1]
        tree_X=self.tree_feats(X,concat=False)
        self.pca_feats = PCA(n_components=n_feats).fit(tree_X)
    
    def __call__(self,X,concat=True):
        tree_X=self.tree_feats(X,concat=False)
        new_X=self.pca_feats.transform(tree_X)
        if(concat):
            new_X=np.concatenate([X,new_X],axis=1)
        return new_X

class ThresExtractor(object):
    def __init__(self,tree_factory=None):
        if(tree_factory is None):
            tree_factory=gradient_tree
        self.tree_factory=tree_factory
        self.thres_feats=None

    def fit(self,X,y):
        tree=self.tree_factory()
        tree.fit(X,y)
        self.thres_feats=tree_feats.make_thres_feats(tree)
#        self.thres_feats.group()

    def __call__(self,X,concat=True):
        return self.thres_feats(X,concat)

def eval_features(in_path):
    clfs=[ tree_feats.GradientTree(),
           tree_feats.RandomTree(),
           { "tree_type":"random", 
             "extract_feats":"basic",
             "clf_type":"SVM",
             "concat":False},
           { "tree_type":"random", 
             "extract_feats":"basic",
             "clf_type":"SVM",
             "concat":True},
           { "tree_type":"random", 
             "extract_feats":"thres",
             "clf_type":"SVM",
             "concat":False},
           { "tree_type":"random", 
             "extract_feats":"thres",
             "clf_type":"SVM",
             "concat":True},

          "LR",
          "SVM"]
    compare_clf(in_path,clfs)

def compare_clf(in_path,clfs):
    data_split=base.get_splits(data_path=in_path,
                               n_splits=10,
                               n_repeats=1)
    for clf_factory_i in clfs:
        if(type(clf_factory_i)==str):
            desc_i=clf_factory_i
            clf_factory_i=base.ClasicalClfFactory(clf_factory_i)
        elif(type(clf_factory_i)==dict):
            names=list(clf_factory_i.keys())
            names.sort()
            desc=[str(clf_factory_i[name_i]) for name_i in names]
            desc_i=",".join(desc)
            clf_factory_i=TreeFeatFactory(clf_factory_i)
        else:
            desc_i=str(clf_factory_i)
        acc_i,balance_i=[],[]
        for clf_j,result_j in data_split.eval(clf_factory_i):
            acc_i.append(result_j.get_acc())
            balance_i.append(result_j.get_metric("balance"))
        print(f"{desc_i}:{np.mean(acc_i):.4f},{np.mean(balance_i):.4f}")

eval_features("bad_exp/data/wine-quality-red")