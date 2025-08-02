import numpy as np
from scipy.stats import entropy 
import base,dataset,tree_feats

class TreeFeatFactory(object):
    def __init__(self,arg_dict):
        self.arg_dict=arg_dict

    def __call__(self):
        return TreeFeatClf(**self.arg_dict)

class TreeFeatClf(object):
    def __init__(self,
                 tree_factory,
                 extr_factory,
                 clf_type,
                 concat=False):
        if(type(tree_factory)==str):
            tree_factory=tree_feats.get_tree(tree_factory)
        if(type(extr_factory)==str):
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

def get_extractor(extr_feats):
    if(extr_feats=="info"):
        return InfoFactory()
    if(extr_feats=="disc"):
        return DiscreteFactory()
    if(extr_feats=="cs"):
        return CSFactory()

class InfoFactory(object):
    def __call__(self,X,y,tree_factory):
        tree=tree_factory()
        tree.fit(X,y)
        tree_dict=tree_feats.make_tree_dict(tree)
        s_feats=tree_feats.inf_features(tree_dict,n_feats=10)
        thres=tree_dict.get_attr("threshold",s_feats)
        feats=tree_dict.get_attr("feat",s_feats)
        return tree_feats.TreeFeatures(features=feats,
                                   thresholds=thres)

class DiscreteFactory(object):
    def __call__(self,X,y,tree_factory):
        tree=tree_factory()
        tree.fit(X,y)
        tree_dict=tree_feats.make_tree_dict(tree)
        s_feats=tree_feats.inf_features(tree_dict,n_feats=20)        
        thres=tree_dict.get_attr("threshold",s_feats)
        feats=tree_dict.get_attr("feat",s_feats)
        return tree_feats.make_disc_feat(feats,thres)

class CSFactory(object):
    def __call__(self,X,y,tree_factory):
        data=dataset.Dataset(X,y)
        n_cats=int(max(y)+1)
        cs_feats=[]
        for i in range(n_cats):
            data_i=data.binarize(i)
            tree_i=tree_factory()
            tree_i.fit(data_i.X,data_i.y)
            tree_dict=tree_feats.make_tree_dict(tree_i)
            s_feats=tree_feats.inf_features(tree_dict,n_feats=10)        
            thres=tree_dict.get_attr("threshold",s_feats)
            feats=tree_dict.get_attr("feat",s_feats)
            feats_i=tree_feats.make_disc_feat(feats,thres)
            cs_feats.append(feats_i)
        def helper(X,concat=True):
            new_X=[cs_i(X) for cs_i in cs_feats]
            new_X=np.concatenate(new_X,axis=1)
            if(concat):
                return np.concatenate([X,new_X],axis=1)
            return new_X
        return helper


 

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    data.y= data.y.astype(int)
    clf=tree_feats.get_tree("random")()
    clf.fit(data.X,data.y)
#    raise Exception(clf.tree_.__getstate__()['nodes'])
    tree_dict=make_tree_dict(clf)
    inf_features(tree_dict)
#    print(tree_dict.get_node(index[:10])[0])
#    print(tree_dict.get_attr("feat",index[:10]))