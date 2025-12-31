import numpy as np
import sklearn.tree
from tabpfn import TabPFNClassifier
import itertools
import base,tree_dict

def get_factory(factory_type):
    if(factory_type=="TabPF"):
        return TabPFFactory
    if(factory_type=="TreeTabPF"):
        return TreeFactory
    if(factory_type=="TreeEnsTabPF"):
        return TreeEnsFactory
    raise Exception(f"Unknow type {factory_type}")

def get_random_tree():
    return sklearn.tree.DecisionTreeClassifier(max_features='sqrt')

class TabPFFactory(base.AbstractClfFactory):
    def __init__(self,feature_params=None):
        pass

    def __call__(self):
        raw_clf=TabPFNClassifier()
        return TabPFAdapter(raw_clf)
    
    def get_info(self):
        return {"clf_type":"TabPF"}

    def __str__(self):
        return "TabPF"

class TabPFAdapter(base.AbstractClfAdapter):
    def __init__(self,tab_model):
        self.tab_model=tab_model
    
    def predict(self,X):
        return self.tab_model.predict(X)
    
    def fit(self,X,y):
        self.tab_model.fit(X,y)    

    def eval(self,data,split_i):
        raise NotImplementedError()

    def save(self,out_path):
        with open(out_path, 'wb') as f:
            pickle.dump(self.tab_model, f)

    def __str__(self):
        return "TabPF"

class TreeFactory(base.AbstractClfFactory):
    def __init__( self,
                  feature_params=None):
        if(feature_params is None):
            feature_params={'feat_type': 'info', 
                            'n_feats': 20}
        self.feature_params=feature_params

    def __call__(self):
        extractor_factory=get_feat_factory(self.feature_params)
        return TreeClf(extractor_factory=extractor_factory)
    
    def get_info(self):
        return {"clf_type":"TREE",
                "feature_params":self.feature_params}

    def __str__(self):
        return "TreeTabPF(optim)"

class TreeClf(base.AbstractClfAdapter):
    def __init__(self,extractor_factory):
        self.extractor_factory=extractor_factory
        self.tab_model=None
        self.extractor=None
    
    def fit(self,X,y):
        self.tab_model=TabPFNClassifier()
        self.extractor=self.extractor_factory(X,y)
        new_X=self.extractor(X)
        self.tab_model.fit(new_X,y)  
    
    def predict(self,X):
        new_X=self.extractor(X)
        return self.tab_model.predict(new_X)

    def eval(self,data,split_i):
        raise NotImplementedError()

    def save(self,out_path):
        utils.make_dir(out_path)
        self.extractor.extractor.save(out_path)

    def __str__(self):
        return "TreeTabPF"

class TreeEnsFactory(base.AbstractClfFactory):
    def __init__( self,
                  feature_params=None):
        if(feature_params is None):
            feature_params={'feat_type': 'info', 
                            'n_feats': 20,
                            'n_clfs':2}
        self.feature_params=feature_params

    def __call__(self):
        extractor_factory=get_feat_factory(self.feature_params)
        return TreeEns(extractor_factory=extractor_factory,
                       n_cls=feature_params['n_clfs'])
    
    def get_info(self):
        return {"clf_type":"TREE-ENS",
                "feature_params":self.feature_params}

    def __str__(self):
        return "TreeEnsTabPF"

class TreeEns(base.AbstractClfAdapter):
    def __init__( self,
                  extractor_factory,
                  n_cls=2):
        self.extractor_factorys=extractor_factory
        self.n_cls=n_cls
        self.models=[]
        self.extractors=[]
    
    def fit(self,X,y):
        for i in range(self.n_cls):
            tab_i=TabPFNClassifier()
            extractor_i=self.extractor_factory(X,y)
            new_X=extractor_i(X)
            tab_i.fit(new_X,y)
            self.extractors.append(extractor_i)
            self.models.append(tab_i) 
#        self.tab_model=TabPFNClassifier()
#        self.extractor=self.extractor_factory(X,y)
#        new_X=self.extractor(X)
#        self.tab_model.fit(new_X,y)  
    
    def predict(self,X):
        all_preds=[]
        for i in range(self.n_cls):
            new_X=self.extractors[i]()
            pred_i=self.models[i].predict(new_X)
        all_preds=np.array(all_preds)
        votes=np.sum(all_preds,axis=1)
        pred=np.argmax(votes,axis=0)
        return pred
 #       new_X=self.extractor(X)
 #       return self.tab_model.predict(new_X)

    def eval(self,data,split_i):
        raise NotImplementedError()

    def save(self,out_path):
        utils.make_dir(out_path)
        self.extractor.extractor.save(out_path)

    def __str__(self):
        return "TreeTabPF"

def get_feat_factory(params_dict):
    n_feats=params_dict["n_feats"]
    feat_type=params_dict['feat_type']
    if(feat_type=="info"):
        feats_factory=InfoFactory
    elif(feat_type=="ind"):
        feats_factory=IndFactory
    elif(feat_type=="prod"):
        feats_factory=ProductFactory
    return feats_factory(n_feats)

class FeatFactory(object):
    def __init__(self,n_feats=10):
        self.n_feats=n_feats

    def __call__(self,X,y):
        tree=get_random_tree()
        tree.fit(X,y)
        t_dict=tree_dict.make_tree_dict(tree)
        return self.make_feats(t_dict)

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
        return TreeFeatures(features=feats,
                            thresholds=thres)        
    @classmethod
    def read(cls,in_path):    
        feats=np.load(f"{in_path}/feats.npy")
        thres=np.load(f"{in_path}/thresholds.npy")
        return TreeFeatures(feats,thres)

class IndFactory(FeatFactory):
    def make_feats(self,tree_dict):
        info,nodes=tree_dict.mutual_info()
        info_index=np.argsort(info)[:self.n_feats]
        s_feats=[nodes[i] for i in info_index]
        thres=tree_dict.get_attr("threshold",s_feats)
        feats=tree_dict.get_attr("feat",s_feats)
        return TreeFeatures(features=feats,
                            thresholds=thres)
    @classmethod
    def read(cls,in_path):    
        feats=np.load(f"{in_path}/feats.npy")
        thres=np.load(f"{in_path}/thresholds.npy")
        return TreeFeatures(feats,thres)

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
        return ProductFeatures( features=feats,
                                thresholds=thres,
                                paths=paths)
    @classmethod
    def read(cls,in_path:str):
        feats=np.load(f"{in_path}/feats.npy")
        thres=np.load(f"{in_path}/thresholds.npy")
        paths=utils.read_json(f"{in_path}/paths")
        return ProductFeatures(feats,thres,paths)

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

    def save(self,out_path):
        utils.make_dir(out_path)
        np.save(f"{out_path}/feats.npy",self.features)
        np.save(f"{out_path}/thresholds.npy",self.thresholds)


class ProductFeatures(TabFeatures):
    def __init__(self,features,
                      thresholds,
                      paths):
        self.features=features
        self.thresholds=thresholds
        self.paths=paths

    def n_feats(self):
        return len(self.paths)

    def compute_feats(self,x_i):
        new_feats=[]
        for i,path_i in enumerate(self.paths):
            values=[]
            for j in path_i:
                value_j = x_i[self.features[j]] 
                thre_j = self.thresholds[j]
                values.append(value_j < thre_j)
            new_feats.append(all(values))
        return np.array(new_feats)

    def save(self,out_path):
        utils.make_dir(out_path)
        np.save(f"{out_path}/feats.npy",self.features)
        np.save(f"{out_path}/thresholds.npy",self.thresholds)
        utils.save_json(self.paths,f"{out_path}/paths")