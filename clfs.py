import numpy as np
import tensorflow as tf
import tree_feats,tree_clf
import base,dataset,deep,utils

def get_clfs(clf_type):
    if(clf_type in base.OTHER_CLFS):
        return base.ClasicalClfFactory(clf_type)
    if(clf_type=="MLP"):
        return MLPFactory()
    if(clf_type=="TREE-MLP"):
        return TreeMLPFactory()
    raise Exception(f"Unknown clf type:{clf_type}")

def basic_callback():
    return tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                            patience=15)

def default_hyperparams():
    return {'layers':2, 'units_0':2,
            'units_1':1,'batch':False}

class NeuralClfFactory(base.AbstractClfFactory):
    def __init__(self,hyper_params=None):
        if(hyper_params is None):
            hyper_params=default_hyperparams()
        self.params=None
        self.hyper_params=hyper_params
    
    def init(self,data):
        class_dict=dataset.get_class_weights(data.y)
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'n_epochs':1000,
                     "class_weights":class_dict}

class NeuralClfAdapter(base.AbstractClfAdapter):
    def __init__(self, params,
                       hyper_params,
                       model=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.model = model
        self.verbose=verbose

    def predict(self,X):
        y=self.model.predict(X,
                             verbose=self.verbose)
        return np.argmax(y,axis=1)

    def predict_proba(self,X):
        return self.model.predict(X,
                             verbose=self.verbose)
    
    def eval(self,data,split_i):
        test_data_i=data.selection(split_i.test_index)
        raw_partial_i=self.predict(test_data_i.X)
        result_i=dataset.Result(y_true=test_data_i.y,
                                y_pred=raw_partial_i)
        return result_i

class MLPFactory(NeuralClfFactory):
    def __call__(self):
        return MLP(params=self.params,
                  hyper_params=self.hyper_params)
    
    def read(self,model_path):
        model_i=tf.keras.models.load_model(model_path)
        clf_i=self()
        clf_i.model=model_i
        return clf_i

    def get_info(self):
        return {"clf_type":"MLP","callback":"basic","hyper":self.hyper_params}

class MLP(NeuralClfAdapter):

    def fit(self,X,y):
        if(self.model is None):
            self.model=deep.single_builder(params=self.params,
                                           hyper_params=self.hyper_params)
        y=tf.one_hot(y,depth=self.params['n_cats'])
        return self.model.fit(x=X,
                              y=y,
                              epochs=self.params['n_epochs'],
                              callbacks=basic_callback(),
                              verbose=self.verbose)

    def save(self,out_path):
        self.model.save(out_path) 

    def __str__(self):
        return "MLP"

class TreeMLPFactory(NeuralClfFactory):
    def __init__(self,
                 hyper_params,
                 tree_params=None):
        if(tree_params is None):
            tree_params={"clf_factory":"random",
                        "extr_factory":("info",30),
                        "concat":True}
        self.hyper_params=hyper_params
        self.tree_params=tree_params

    def __call__(self):
        return TreeMLP(params=self.params,
                       hyper_params=self.hyper_params,
                       tree_features=TreeFeatures(**self.tree_params))
    
#    def read(self,model_path):
#        model_i=tf.keras.models.load_model(f"{model_path}/nn")
#        tree_repr=np.load(f"{model_path}/tree.npy")
#        nodes=np.load(f"{model_path}/nodes.npy")
#        tree=TreeFeatures(tree_repr,nodes)
#        clf_i= TreeMLP(params=self.params,
#                       hyper_params=self.hyper_params,
#                       model=(model_i,tree))
#        return clf_i
    
    def get_info(self):
        return {"clf_type":"TREE-MLP","callback":"basic",
                "hyper":self.hyper_params,
                "extr_dict":self.extr_dict}

class TreeFeatures(object):
    def __init__(self,clf_factory,
                      extr_factory,
                      concat,
                      ens_type=None):
        if(type(clf_factory)==str):
            clf_factory=tree_feats.get_tree(clf_factory)
        if(type(extr_factory)==tuple):
            extr_factory=tree_clf.get_extractor(extr_factory)
        self.clf_factory=clf_factory
        self.extr_factory=extr_factory
        self.concat=concat
        self.ens_type=ens_type
    
    def __call__(self,X,y):
        extr=self.extr_factory(X,y,self.clf_factory)
        return ExctractorCurry(extr,self.concat)

class ExctractorCurry(object):
    def __init__(self,extractor,concat):
        self.extractor=extractor
        self.concat=concat

    def __call__(self,X):
        return self.extractor(X,self.concat)

class TreeMLP(NeuralClfAdapter):
    def __init__(self, params,
                       hyper_params,
                       tree_features,
                       model=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.tree_features=tree_features
        self.extractor=None
        self.model = model
        self.verbose=verbose

    def fit(self,X,y):
        self.extractor=self.tree_features.gen(X,y)
        new_X=self.extractor(X)
        if(self.model is None):
            params_i=self.params.copy()
            params_i["dims"]=(new_X.shape[1],)
            self.model=MLP(params=params_i,
                           hyper_params=self.hyper_params)
        self.model.fit(new_X,y)
    
    def predict(self,X):
        new_X=self.extractor(X)
        return self.model.predict(new_X)
    
#    def save(self,out_path):
#        utils.make_dir(out_path)
#        np.save(f"{out_path}/tree.npy",self.tree.tree_repr)
#        np.save(f"{out_path}/nodes.npy",self.tree.selected_nodes)
#        self.model.save(f"{out_path}/nn.keras") 

class CSTreeEnsFactory(NeuralClfFactory):
    def __init__(self,
                 hyper_params,
                 tree_params=None):
        if(tree_params is None):
            extr_params={"clf_factory":"random",
                        "extr_factory":("info",30),
                        "concat":True,"ens_type":"binary"}
            ens_params={"ens_type":"binary"}
        else:
            keys=["clf_factory","extr_factory","concat"]
            extr_params,ens_params=utils.split_dict(tree_params,
                                                    keys)
        self.hyper_params=hyper_params
        self.extr_params=extr_params
        self.ens_params=ens_params

    def __call__(self):
        tree_features=TreeFeatures(**self.extr_params)
        return CSTreeEns(params=self.params,
                         hyper_params=self.hyper_params,
                         tree_features=tree_features,
                         extr_gen=self.get_extr_gen(),
                         weight_gen=self.get_weight_gen())

    def get_extr_gen(self):
        if(self.ens_params["ens_type"]=="binary"):
            return binary_gen
        else:
            return full_gen

    def get_weight_gen(self):
        if(self.ens_params["weights"]=="specific"):
            return cs_weights
        return basic_weights

    def get_info(self):
        return {"clf_type":"CSTreeEns","callback":"basic",
                "hyper":self.hyper_params,
                "extr_dict":self.extr_dict}

class CSTreeEns(NeuralClfAdapter):
    def __init__(self, params,
                       hyper_params,
                       tree_features,
                       extr_gen,
                       weight_gen,
                       model=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.tree_features=tree_features
        self.extr_gen=extr_gen
        self.weight_gen=weight_gen
        self.model = model
        self.all_extract=[]
        self.all_clfs=[]
        self.verbose=verbose

    def fit(self,X,y):
        gen=self.extr_gen(X,y,self.tree_features)
        for i,extr_i in enumerate(gen):
            new_X=extr_i(X)
            self.all_extract.append(extr_i)
            params_i=self.params.copy()
            params_i["dims"]=(new_X.shape[1],)
            clf_i=MLP(params=self.weight_gen(i,params_i),
                      hyper_params=self.hyper_params)
            clf_i.fit(X=new_X,
                      y=y)
            self.all_clfs.append(clf_i)

    def predict(self,X):
        votes=[]
        for i,extr_i in enumerate(self.all_extract):
            X_i=extr_i(X=X)#,
#                      concat=self.tree_features.concat)
            y_i=self.all_clfs[i].predict(X_i)
            votes.append(y_i)
        votes=np.array(votes,dtype=int)
        y_pred=[]
        for vote_i in votes.T:
            counts=np.bincount(vote_i)
            y_pred.append(np.argmax(counts))
        return y_pred


def full_gen(self,X,y,tree_features):
    n_iters=int(max(y)+1)
    for _ in range(n_iters):
        yield tree_features(X=X,y=y)

def binary_gen(X,y,tree_features):
    data=dataset.Dataset(X,y)
    n_cats=int(max(y)+1)
    for i in range(n_cats):
        data_i=data.binarize(i)
        yield tree_features(X=data_i.X,
                            y=data_i.y)

def basic_weights(i,params):
    return params

def cs_weights(i,params):
    params["class_weights"][i]*=2
    return params