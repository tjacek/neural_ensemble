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
                 tree_type="random",
                 feat_type="info",
                 n_feats=30,
                 concat=True):
        self.hyper_params=hyper_params
        self.extr_dict={"clf_factory":tree_type,
                        "extr_factory":(feat_type,n_feats),
                        "concat":"concat"}

    def __call__(self):
        return TreeMLP(params=self.params,
                       hyper_params=self.hyper_params,
                       tree_features=TreeFeatures(**self.extr_dict))
    
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
                      concat):
        if(type(clf_factory)==str):
            clf_factory=tree_feats.get_tree(clf_factory)
        if(type(extr_factory)==tuple):
            extr_factory=tree_clf.get_extractor(extr_factory)
        self.clf_factory=clf_factory
        self.extr_factory=extr_factory
        self.concat=concat
        self.extractor=None
    
    def fit(self,X,y):
        self.extractor=self.extr_factory(X,y,self.clf_factory)

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
        self.model = model
        self.verbose=verbose

    def fit(self,X,y):
        self.tree_features.fit(X,y)
        new_X=self.tree_features(X)
        if(self.model is None):
            params_i=self.params.copy()
            dims=params_i["dims"]
            params_i["dims"]=(new_X.shape[1],)
            self.model=MLP(params=params_i,
                           hyper_params=self.hyper_params)
        self.model.fit(new_X,y)
    
    def predict(self,X):
        new_X=self.tree_features(X)
        return self.model.predict(new_X)
    
#    def save(self,out_path):
#        utils.make_dir(out_path)
#        np.save(f"{out_path}/tree.npy",self.tree.tree_repr)
#        np.save(f"{out_path}/nodes.npy",self.tree.selected_nodes)
#        self.model.save(f"{out_path}/nn.keras") 