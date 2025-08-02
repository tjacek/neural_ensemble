import numpy as np
from scipy.stats import entropy 
import base,dataset,tree_feats

class TreeFeatFactory(object):
    def __init__(self,arg_dict):
        self.arg_dict=arg_dict

    def __call__(self):
        return TreeFeatClf(**self.arg_dict)

    def __str__(self):
        names=list(self.arg_dict.keys())
        names.sort()
        desc=[str(self.arg_dict[name_i]) 
                for name_i in names]
        return ",".join(desc)

class TreeFeatClf(object):
    def __init__(self,
                 tree_factory,
                 extr_factory,
                 clf_type,
                 concat=False):
        if(type(tree_factory)==str):
            tree_factory=tree_feats.get_tree(tree_factory)
        if(type(extr_factory)==str or 
            type(extr_factory)==tuple):
            extr_factory=get_extractor(extr_factory)
        self.tree_factory=tree_factory
        self.extr_factory=extr_factory
        self.clf_type=clf_type
        self.concat=concat
        self.extractor=None
        self.clf=None

    def fit(self,X,y):
        self.extractor=self.extr_factory(X,y,
                                         self.tree_factory)
        self.clf=base.get_clf(self.clf_type)
        X=self.extractor(X=X,
                         concat=self.concat)
        self.clf.fit(X,y)

    def predict(self,X):
        X=self.extractor(X=X,
                         concat=self.concat)
        return self.clf.predict(X)

class TreeEns(object):
    def __init__(self,
                 tree_factory,
                 extr_factory,
                 clf_type,
                 concat=False):
        self.tree_factory=tree_factory
        self.extr_factory=extr_factory
        self.clf_type=clf_type
        self.concat=concat
        self.all_extract=[]
        self.all_clfs=[]

    def fit(self,X,y):
        data=dataset.Dataset(X,y)
        n_cats=int(max(y)+1)
        for i in range(n_cats):
            data_i=data.binarize(i)
            extr_i=self.extr_factory(X=data_i.X,
                                     y=data_i.y,
                                     tree_factory=self.tree_factory)           
            self.all_extract.append(extr_i)
            X_i=extr_i(X=X,
                         concat=self.concat)
            clf_i=base.get_clf(self.clf_type)
            clf_i.fit(X_i,y)
            self.all_clfs.append(clf_i)

    def predict(self,X):
        all_pred=[]
        for i,extr_i in enumerate(self.all_extract):
            X_i==extr_i(X=X,concat=self.concat)
            y_i=self.all_clfs[i].predict(X_i)
            all_pred.append(y_i)
        all_pred=np.array(all_pred)
        raise Exception()          

def get_extractor(extr_feats):
    if(type(extr_feats)==tuple):
        extr_type,n_feats=extr_feats
        factory=get_extr_type(extr_type)
        return factory(n_feats)
    extr_type=get_extr_type(extr_feats)
    return extr_type()

def get_extr_type(extr_feats):
    if(extr_feats=="info"):
        return InfoFactory
    if(extr_feats=="disc"):
        return DiscreteFactory
    if(extr_feats=="ind"):
        return IndFactory
    if(extr_feats=="cs"):
        return CSFactory

class InfoFactory(object):
    def __init__(self,n_feats=10):
        self.n_feats=n_feats

    def __call__(self,X,y,tree_factory):
        tree=tree_factory()
        tree.fit(X,y)
        tree_dict=tree_feats.make_tree_dict(tree)
        s_feats=tree_feats.inf_features(tree_dict,
                                        n_feats=self.n_feats)
        thres=tree_dict.get_attr("threshold",s_feats)
        feats=tree_dict.get_attr("feat",s_feats)
        return tree_feats.TreeFeatures(features=feats,
                                   thresholds=thres)

class DiscreteFactory(object):
    def __init__(self,n_feats=20):
        self.n_feats=n_feats

    def __call__(self,X,y,tree_factory):
        tree=tree_factory()
        tree.fit(X,y)
        tree_dict=tree_feats.make_tree_dict(tree)
        s_feats=tree_feats.inf_features(tree_dict,
                                        n_feats=self.n_feats)        
        thres=tree_dict.get_attr("threshold",s_feats)
        feats=tree_dict.get_attr("feat",s_feats)
        return tree_feats.make_disc_feat(feats,thres)

class IndFactory(object):
    def __init__(self,n_feats=20):
        self.n_feats=n_feats

    def __call__(self,X,y,tree_factory):
        tree=tree_factory()
        tree.fit(X,y)
        tree_dict=tree_feats.make_tree_dict(tree)
        mutual_info=tree_dict.mutual_info()
        index=np.argsort(mutual_info)
        s_nodes=index[:self.n_feats]
        return tree_feats.IndFeatures(s_nodes,tree)

class CSFactory(object):
    def __init__(self,n_feats=10):
        self.n_feats=n_feats

    def __call__(self,X,y,tree_factory):
        data=dataset.Dataset(X,y)
        n_cats=int(max(y)+1)
        cs_feats=[]
        for i in range(n_cats):
            data_i=data.binarize(i)
            tree_i=tree_factory()
            tree_i.fit(data_i.X,data_i.y)
            tree_dict=tree_feats.make_tree_dict(tree_i)
            s_feats=tree_feats.inf_features(tree_dict,
                                            n_feats=self.n_feats)        
            thres=tree_dict.get_attr("threshold",s_feats)
            feats=tree_dict.get_attr("feat",s_feats)
            feats_i=tree_feats.make_disc_feat(feats,thres)
            cs_feats.append(feats_i)
        return tree_feats.ConcatFeatures(cs_feats) 

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    data.y= data.y.astype(int)
    clf=tree_feats.get_tree("random")()
    clf.fit(data.X,data.y)
    tree_dict=make_tree_dict(clf)
    inf_features(tree_dict)