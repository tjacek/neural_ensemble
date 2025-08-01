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
        tree_dict=make_tree_dict(tree)
        s_feats=inf_features(tree_dict,n_feats=10)
        thres=tree_dict.get_attr("threshold",s_feats)
        feats=tree_dict.get_attr("feat",s_feats)
        return tree_feats.TreeFeatures(features=feats,
                                   thresholds=thres)

class DiscreteFactory(object):
    def __call__(self,X,y,tree_factory):
        tree=tree_factory()
        tree.fit(X,y)
        tree_dict=make_tree_dict(tree)
        s_feats=inf_features(tree_dict,n_feats=20)        
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
            tree_dict=make_tree_dict(tree_i)
            s_feats=inf_features(tree_dict,n_feats=10)        
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

class TreeDict(dict):
    def __len__(self):
        return len(self["feat"])

    def get_node(self,i):
        keys=list(self.keys())
        keys.sort()
        print(keys)
        if(type(i)!=list):
            return [self[key_j][i] for key_j in keys]
        return [[self[key_j][k] for key_j in keys]
                         for k in i]
    
    def get_attr(self,key,nodes):
        return [self[key][i] for i in nodes]

    def mutual_info(self):
        dist=self["value"][0]
        n_samples=len(self)
        offset=np.ones(dist.shape)
        h_y=entropy(dist)
        mi=[]
        for i,value_i in enumerate(self["value"]):
            samples_i=self["samples"][i]
            value_i=self["value"][i]
            p_y= samples_i/ n_samples
            h_yx=p_y*entropy(value_i)#,nan_policy='omit')
            h_yx+=(1-p_y)*entropy(offset-value_i)
            i_xy=h_y-h_yx
            mi.append(i_xy)
        return mi

    def get_path(self,i):
        path=[i]
        while(i>=0):
            i=self["parent"][i]
            path.append(i)
        return path

def make_tree_dict(clf):
    tree_dict=TreeDict()
    tree_dict["threshold"]=clf.tree_.threshold
    tree_dict["feat"]=clf.tree_.feature
    tree_dict["left"]=clf.tree_.children_left
    tree_dict["right"]=clf.tree_.children_right
    tree_dict["value"]=[ value_i.flatten() 
                            for value_i in clf.tree_.value]
    tree_dict["samples"]=clf.tree_.weighted_n_node_samples
    n_nodes=len(tree_dict["feat"])
    tree_dict["parent"]= -np.ones((n_nodes,),dtype=int)
    for i in range(n_nodes):
        left_i=tree_dict["left"][i]
        right_i=tree_dict["right"][i]
        if(left_i>=0):
            tree_dict["parent"][left_i]=i
            tree_dict["parent"][right_i]=i
    return tree_dict

def inf_features(tree_dict,n_feats=5):
    mutual_info=tree_dict.mutual_info()
    index=np.argsort(mutual_info)
    s_feats=[]
    for i in index[:n_feats]:
        s_feats+=tree_dict.get_path(i)
    s_feats=set(s_feats)
    s_feats.remove(-1)
    s_feats=list(set(s_feats))
    return s_feats
 

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