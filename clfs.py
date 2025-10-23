import numpy as np
import tensorflow as tf
import re
import tree_feats,tree_clf
import base,dataset,deep,utils

def get_clfs(clf_type,
             hyper_params,
             feature_params=None):
    if(clf_type in base.OTHER_CLFS):
        return base.ClasicalClfFactory(clf_type)
    if(hyper_params is None):
         hyper_params=default_hyperparams()
    if(clf_type=="MLP"):
        return MLPFactory()
    if(clf_type=="TREE-MLP"):
        return TreeMLPFactory(feature_params=feature_params)
    if(feature_params is None):
        feature_params={ "tree_factory":"random",
                     "extr_factory":("info",30),
                     "concat":True}

    if(clf_type=="TREE-ENS"):
        return CSTreeEnsFactory(hyper_params=hyper_params,
                                feature_params=feature_params,
                                ens_params={ "ens_type":"basic",
                                              "weights":"basic"})
    if(clf_type=="BINARY-TREE-ENS"):
        return CSTreeEnsFactory(hyper_params=hyper_params,
                                feature_params=feature_params,
                                ens_params={ "ens_type":"binary",
                                             "weights":"basic"})
    
    if(clf_type=="CS-TREE-ENS"):
        return CSTreeEnsFactory(hyper_params=hyper_params,
                                feature_params=feature_params,
                                ens_params={ "ens_type":"basic",
                                              "weights":"CS"})
    if(clf_type=="BINARY-CS-TREE-ENS"):
        return CSTreeEnsFactory(hyper_params=hyper_params,
                                feature_params=feature_params,
                                ens_params={ "ens_type":"binary",
                                             "weights":"CS"})
    reg_expr=re.compile("(\\D)+(\\d)+")
    if(reg_expr.match(clf_type)):
        n=utils.extract_number(clf_type)
        return CSTreeEnsFactory(hyper_params=hyper_params,
                                feature_params=feature_params,
                                ens_params={ "ens_type":n,
                                              "weights":"basic"})
    raise Exception(f"Unknown clf type:{clf_type}")   

def basic_callback():
    return tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                            patience=15)

def default_hyperparams():
    return {'layers':1, 'units_0':1,
            'units_1':1,'batch':False}

def get_nn_type(nn_type):
    if(nn_type=="MLP"):
        return MLPFactory
    if(nn_type=="TREE-MLP"):
        return TreeMLPFactory
    if(nn_type=="TREE-ENS"):
        return CSTreeEnsFactory
    raise Exception(f"Unknown clf type:{nn_type}")    

def read_factory(in_path):
    info_dict=utils.read_json(in_path)
    nn_type=info_dict['clf_type']
    factory_type=get_nn_type(nn_type)
    if(nn_type=="TREE-MLP"):
        factory=factory_type(hyper_params=info_dict["hyper"],
                             feature_params=info_dict["feature_params"])
    if(nn_type=="TREE-ENS"):
        factory=factory_type(hyper_params=info_dict["hyper"],
                             feature_params=info_dict["feature_params"],
                             ens_params=info_dict["ens_params"])
    return factory

class FeatureExtactorFactory(object):
    def __init__(self,tree_factory,
                      extr_factory,
                      concat):
        if(type(tree_factory)==str):
            tree_factory=tree_feats.get_tree(tree_factory)
        if(type(extr_factory)==tuple):
            extr_factory=tree_clf.get_extractor(extr_factory)
        self.tree_factory=tree_factory
        self.extr_factory=extr_factory
        self.concat=concat
    
    def __call__(self,X,y):
        extr=self.extr_factory(X,y,self.tree_factory)
        return FeatureExtactor(extr,self.concat)

class FeatureExtactor(object):
    def __init__(self,extractor,concat=True):
        self.extractor=extractor
        self.concat=concat

    def __call__(self,X):
        return self.extractor(X,self.concat)

    def save(self,out_path):
        self.extractor.save(out_path)

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
        return {"clf_type":"MLP","callback":"basic",
                "hyper":self.hyper_params}

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
                 hyper_params=None,
                 feature_params=None):
        if(hyper_params is None):
            hyper_params=default_hyperparams()
        if(feature_params is None):
            feature_params={"tree_factory":"random",
                            "extr_factory":("info",30),
                            "concat":True}
        self.hyper_params=hyper_params
        self.feature_params=feature_params

    def __call__(self):
        extractor_factory=FeatureExtactorFactory(**self.feature_params)
        return TreeMLP(params=self.params,
                       hyper_params=self.hyper_params,
                       extractor_factory=extractor_factory)
    
    def read(self,in_path):
        in_path=in_path.split(".")[0]
        model_i=tf.keras.models.load_model(f"{in_path}/nn.keras")
        extractor=tree_feats.read_feats(f"{in_path}/tree")
        tree_mlp=self()
        tree_mlp.model=model_i
        tree_mlp.extractor=FeatureExtactor(extractor)
        return tree_mlp
    
    def get_info(self):
        return {"clf_type":"TREE-MLP","callback":"basic",
                "hyper":self.hyper_params,
                "feature_params":self.feature_params}

class TreeMLP(NeuralClfAdapter):
    def __init__(self, params,
                       hyper_params,
                       extractor_factory,
                       model=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.extractor_factory=extractor_factory
        self.extractor=None
        self.model = model
        self.verbose=verbose

    def fit(self,X,y):
        self.extractor=self.extractor_factory(X,y)
        new_X=self.extractor(X)
        if(self.model is None):
            params_i=self.params.copy()
            params_i["dims"]=(new_X.shape[1],)
            self.model=MLP(params=params_i,
                           hyper_params=self.hyper_params)
        return self.model.fit(new_X,y)


    def predict(self,X):
        new_X=self.extractor(X)
        return self.model.predict(new_X)#,

    def save(self,out_path):
        utils.make_dir(out_path)
        self.extractor.extractor.save(f"{out_path}/tree")
        self.model.save(f"{out_path}/nn.keras")

    def __str__(self):
        return "TREE-MLP"
        
class CSTreeEnsFactory(NeuralClfFactory):
    def __init__(self,
                 hyper_params,
                 feature_params=None,
                 ens_params=None):
        if(feature_params is None):
            feature_params={ "tree_factory":"random",
                             "extr_factory":("info",30),
                             "concat":True}
        if(ens_params is None):
            ens_params={ "ens_type":"binary",
                         "weights":"basic"}
        self.hyper_params=hyper_params
        self.feature_params=feature_params
        self.ens_params=ens_params
        self.params=None

    def __call__(self):
        extractor_factory=FeatureExtactorFactory(**self.feature_params)
        return CSTreeEns(params=self.params,
                         hyper_params=self.hyper_params,
                         extractor_factory=extractor_factory,
                         extr_gen=self.get_extr_gen(),
                         weight_gen=self.get_weight_gen())

    def get_extr_gen(self):
        ens_type=self.ens_params["ens_type"]
        if(type(ens_type)==int):
            return FullGen(ens_type)
        if(ens_type=="binary"):
            return binary_gen
        else:
            return FullGen()

    def get_weight_gen(self):
        if(self.ens_params["weights"]=="specific"):
            return cs_weights
        return basic_weights
    
    def read(self,in_path):
        tree_ens=self()
        for path_i in utils.top_files(in_path):
            model_i=tf.keras.models.load_model(f"{path_i}/nn.keras")
            tree_ens.all_clfs.append(MLP(None,None,model_i))
            extractor_i=tree_feats.read_feats(f"{path_i}/tree")
            extractor_i=FeatureExtactor(extractor_i,
                                   concat=self.feature_params["concat"])
            tree_ens.all_extract.append(extractor_i)
        return tree_ens

    def get_info(self):
        return {"clf_type":"TREE-ENS","callback":"basic",
                "hyper":self.hyper_params,
                "feature_params":self.feature_params,
                "ens_params":self.ens_params}

class CSTreeEns(NeuralClfAdapter):
    def __init__(self, params,
                       hyper_params,
                       extractor_factory,
                       extr_gen,
                       weight_gen,
                       model=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.extractor_factory=extractor_factory
        self.extr_gen=extr_gen
        self.weight_gen=weight_gen
        self.model = model
        self.all_extract=[]
        self.all_clfs=[]
        self.verbose=verbose

    def fit(self,X,y):
        gen=self.extr_gen(X,y,self.extractor_factory)
        history=[]
        for i,extr_i in enumerate(gen):
            new_X=extr_i(X)
            self.all_extract.append(extr_i)
            params_i=self.params.copy()
            params_i["dims"]=(new_X.shape[1],)
            clf_i=MLP(params=self.weight_gen(i,params_i),
                      hyper_params=self.hyper_params)
            history_i=clf_i.fit(X=new_X,
                                y=y)
            history.append(history_i)
            self.all_clfs.append(clf_i)
        return history

    def predict(self,X):
        votes=[]
        for i,extr_i in enumerate(self.all_extract):
            X_i=extr_i(X=X)#,
            y_i=self.all_clfs[i].predict_proba(X_i)#,
            votes.append(y_i)
        votes=np.array(votes)#,dtype=int)
        votes=np.sum(votes,axis=0)
        y_pred=np.argmax(votes,axis=1)
#        y_pred=[]
#        for vote_i in votes.T:
#            counts=np.bincount(vote_i)
#            y_pred.append(np.argmax(counts))
        return y_pred

    def votes(self,X):
        partial_y=[]
        for i,extr_i in enumerate(self.all_extract):
            X_i=extr_i(X=X)#,
            y_i=self.all_clfs[i].predict_proba(X_i)
            partial_y.append(y_i)
        return np.array(partial_y)
    
    def save(self,out_path):
        utils.make_dir(out_path)
        for i,clf_i in enumerate(self.all_clfs):
            out_i=f"{out_path}/{i}"
            utils.make_dir(out_i)
            extr_i=self.all_extract[i]
            extr_i.save(f"{out_i}/tree")
            clf_i.save(f"{out_i}/nn.keras")

    def __str__(self):
        n_clf=len(self.all_clfs)
        return f"TREE-ENS:{n_clf}"

class FullGen(object):
    def __init__(self,n_iters=None):
        self.n_iters=n_iters

    def __call__(self,X,y,tree_features):
        if(self.n_iters is None):
            self.n_iters=int(max(y)+1)
        for _ in range(self.n_iters):
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