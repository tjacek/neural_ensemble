import numpy as np
from scipy.stats import entropy 
import collections,itertools
import base,dataset,tree_feats,utils

class TreeFeatFactory(object):
    def __init__(self,arg_dict,clf_type="clf"):
        if(clf_type=="clf"):
            clf_cls=TreeFeatClf
        else:
            clf_cls=TreeEns
        self.arg_dict=arg_dict
        self.clf_type=clf_type
        self.clf_cls=clf_cls

    def __call__(self):
        return self.clf_cls(**self.arg_dict)

    def __str__(self):
        names=list(self.arg_dict.keys())
        names.sort()
        desc=[]
        for name_i in names:
            arg_i= self.arg_dict[name_i]
            if(type(arg_i)==tuple):
                arg_i=str(arg_i).replace("(","").replace(")","")
            desc.append(str(arg_i))
        desc=[self.clf_type]+desc
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
                 concat=False,
                 n_iters=None):
        if(type(tree_factory)==str):
            tree_factory=tree_feats.get_tree(tree_factory)
        if(type(extr_factory)==str or 
            type(extr_factory)==tuple):
            extr_factory=get_extractor(extr_factory)
        self.tree_factory=tree_factory
        self.extr_factory=extr_factory
        self.clf_type=clf_type
        self.concat=concat
        self.all_extract=[]
        self.all_clfs=[]
        self.n_iters=n_iters
    
    def is_binary(self):
        return self.n_iters is None
    
    def fit(self,X,y):
        data=dataset.Dataset(X,y)        
        if(self.is_binary()):
            n_cats=int(max(y)+1)
            for i in range(n_cats):
                data_i=data.binarize(i)
                self.add_clf(X,y,data_i)
        else:
            for i in range(self.n_iters):
                self.add_clf(X,y,data)

    def add_clf(self,X,y,data_i):
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
        votes=[]
        for i,extr_i in enumerate(self.all_extract):
            X_i=extr_i(X=X,concat=self.concat)
            y_i=self.all_clfs[i].predict(X_i)
            votes.append(y_i)
        votes=np.array(votes,dtype=int)
        y_pred=[]
        for vote_i in votes.T:
            counts=np.bincount(vote_i)
            y_pred.append(np.argmax(counts))
        return y_pred

def get_extractor(extr_feats):
    if(type(extr_feats)==tuple or 
        type(extr_feats)==list):
        extr_type,n_feats=extr_feats
        factory=get_extr_type(extr_type)
        return factory(n_feats)
    extr_type=get_extr_type(extr_feats)
    return extr_type()

def get_extr_type(extr_feats):
    if(extr_feats=="info"):
        return InfoFactory
#    if(extr_feats=="disc"):
#        return DiscreteFactory
    if(extr_feats=="ind"):
        return IndFactory
    if(extr_feats=="prod"):
        return ProductFactory
    if(extr_feats=="cs"):
        return CSFactory
    if(extr_feats=="mixed"):
        return MixedFactory
    raise Exception(f"Unknown extr type:{extr_feats}")

class FeatFactory(object):
    def __init__(self,n_feats=10):
        self.n_feats=n_feats

    def __call__(self,X,y,tree_factory):
        tree=tree_factory()
        tree.fit(X,y)
        tree_dict=tree_feats.make_tree_dict(tree)
        return self.make_feats(tree_dict)

    def make_feats(self,tree_feats):
        raise NotImplementedError()

class InfoFactory(FeatFactory):
    def make_feats(self,tree_dict):
        info,nodes=tree_dict.mutual_info()
        info_index=np.argsort(info)[:self.n_feats]
        s_nodes=[nodes[i] for i in info_index]
        s_feats=tree_dict.get_paths(s_nodes)
        thres=tree_dict.get_attr("threshold",s_feats)
        feats=tree_dict.get_attr("feat",s_feats)
        return tree_feats.TreeFeatures(features=feats,
                                       thresholds=thres)        
    @classmethod
    def read(cls,in_path):    
        feats=np.load(f"{in_path}/feats.npy")
        thres=np.load(f"{in_path}/thresholds.npy")
        return tree_feats.TreeFeatures(feats,thres)

class IndFactory(FeatFactory):
    def make_feats(self,tree_dict):
        info,nodes=tree_dict.mutual_info()
        info_index=np.argsort(info)[:self.n_feats]
        s_feats=[nodes[i] for i in info_index]
        thres=tree_dict.get_attr("threshold",s_feats)
        feats=tree_dict.get_attr("feat",s_feats)
        return tree_feats.TreeFeatures(features=feats,
                                       thresholds=thres)
    @classmethod
    def read(cls,in_path):    
        feats=np.load(f"{in_path}/feats.npy")
        thres=np.load(f"{in_path}/thresholds.npy")
        return tree_feats.TreeFeatures(feats,thres)

class ProductFactory(FeatFactory):
    def make_feats(self,tree_dict):
        info,nodes=tree_dict.mutual_info()
        info_index=np.argsort(info)[:self.n_feats]
        s_nodes=[nodes[i] for i in info_index]
        indv_paths=list(tree_dict.indv_paths(s_nodes))
        path_nodes = list(itertools.chain.from_iterable(indv_paths))
        path_nodes=list(set(path_nodes))
        thres=tree_dict.get_attr("threshold",path_nodes)
        feats=tree_dict.get_attr("feat",path_nodes)
        node_dict={ node_i:i for i,node_i in enumerate(path_nodes)}
        paths=[[ node_dict[j] for j in path_i]
                    for path_i in indv_paths]
        return tree_feats.ProductFeatures(features=feats,
                                          thresholds=thres,
                                          paths=paths)
    @classmethod
    def read(cls,in_path:str):
        feats=np.load(f"{in_path}/feats.npy")
        thres=np.load(f"{in_path}/thresholds.npy")
        paths=utils.read_json(f"{in_path}/paths")
        return tree_feats.ProductFeatures(feats,thres,paths)

#class DiscreteFactory(object):
#    def __init__(self,n_feats=20):
#        self.n_feats=n_feats

#    def __call__(self,X,y,tree_factory):
#        tree=tree_factory()
#        tree.fit(X,y)
#        tree_dict=tree_feats.make_tree_dict(tree)
#        s_feats=tree_feats.inf_features(tree_dict,
#                                        n_feats=self.n_feats)        
#        thres=tree_dict.get_attr("threshold",s_feats)
#        feats=tree_dict.get_attr("feat",s_feats)
#        return tree_feats.make_disc_feat(feats,thres)

#class IndFactory(object):
#    def __init__(self,n_feats=20):
#        self.n_feats=n_feats

#    def __call__(self,X,y,tree_factory):
#        tree=tree_factory()
#        tree.fit(X,y)
#        tree_dict=tree_feats.make_tree_dict(tree)
#        mutual_info=tree_dict.mutual_info()
#        index=np.argsort(mutual_info)
#        s_nodes=index[:self.n_feats]
#        return tree_feats.IndFeatures(s_nodes,tree)

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

class MixedFactory(object):
    def __init__(self,n_feats=30,n_trees=10):
        self.n_feats=n_feats
        self.n_trees=n_trees
        
    def __call__(self,X,y,tree_factory):
        tree_dicts=self.get_dicts(X,y,tree_factory)
        nodes_by_tree=self.get_nodes(tree_dicts)
        for i,s_nodes in nodes_by_tree.items():
            tree_i=tree_dicts[i]
            print(tree_i)
        raise Exception(nodes_by_tree)

    def get_dicts(self,X,y,tree_factory):
        tree_dicts=[]
        for i in range(self.n_trees):
            tree_i=tree_factory()
            tree_i.fit(X,y)
            tree_dict_i=tree_feats.make_tree_dict(tree_i)
            tree_dicts.append(tree_dict_i)  
        return tree_dicts

    def get_nodes(self,tree_dicts):
        MixedNode = collections.namedtuple('MixedNode', 'tree node mutual')
        mixed_nodes=[]
        for i,tree_dict_i in enumerate(tree_dicts):
            mutual_i=tree_dict_i.mutual_info()
            for j,mutual_j in enumerate(mutual_i):
                node_j=MixedNode(i,j,mutual_j)
                mixed_nodes.append(node_j)
        mixed_nodes = sorted(mixed_nodes, key=lambda x: x.mutual, reverse=True)
        s_nodes=mixed_nodes[:self.n_feats]
        nodes_by_tree= collections.defaultdict(lambda:[])
        for node_i in s_nodes:
            nodes_by_tree[node_i.tree].append(node_i.node)
        return nodes_by_tree

class DistGroup(object):
    def __init__(self,base_dist):
        self.all_dist=[base_dist]

    def add_dist(self,dist):
        self.all_dist.append(dist)

    def __call__(self,dist):
        return np.sum([mutual_info(dist,dist_i) 
                        for dist_i in self.all_dist])

if __name__ == '__main__':
    import base,dataset
    data=dataset.read_csv("bad_exp/data/wine-quality-red")
    data.y= data.y.astype(int)
    clf=tree_feats.get_tree("random")()
    clf.fit(data.X,data.y)
    tree_dict=make_tree_dict(clf)
    inf_features(tree_dict)