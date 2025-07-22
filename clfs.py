import numpy as np
import tensorflow as tf
import base,dataset,deep

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

    def predict(self,X):
        y=self.model.predict(X,
                             verbose=self.verbose)
        return np.argmax(y,axis=1)

    def predict_proba(self,X):
        return self.model.predict(X,
                             verbose=self.verbose)

    def save(self,out_path):
        self.model.save(out_path) 

    def eval(self,data,split_i):
        test_data_i=data.selection(split_i.test_index)
        raw_partial_i=self.predict(test_data_i.X)
        result_i=dataset.Result(y_true=test_data_i.y,
                                y_pred=raw_partial_i)
        return result_i

    def __str__(self):
        return "MLP"

class TreeMLPFactory(NeuralClfFactory):
    def __call__(self):
        return TreeMLP(params=self.params,
                       hyper_params=self.hyper_params)

class TreeMLP(NeuralClfAdapter):
    def __init__(self, params,
                       hyper_params,
                       model=None,
                       verbose=0):

        if(model is None):
            nn_model,tree=None,None
        else:
            nn_model,tree=model
        self.params=params
        self.hyper_params=hyper_params
        self.model = nn_model
        self.tree=tree
        self.verbose=verbose

    def fit(self,X,y):
        tree=base.get_clf("TREE")
        tree.fit(X,y)
        self.tree=make_tree_features(tree)
        new_X=self.tree(X)
        if(self.model is None):
            self.model=MLP(params=self.params,
                           hyper_params=self.hyper_params)
        raise Exception(new_X.shape)

class TreeFeatures(object):
    def  __init__(self,tree_repr,
                       selected_nodes):
        self.tree_repr=tree_repr
        self.selected_nodes=selected_nodes

    def __call__(self,X,concat=True):
        new_feats=[self.compute_feats(x_i) for x_i in X]
        new_feats=np.array(new_feats)
        if(concat):
            return np.concatenate([X,new_feats],axis=1)
        return new_feats

    def compute_feats(self,x_i):
        new_feats=[]
        for i in self.selected_nodes:
            node_i=self.tree_repr[i]
            feat_index,thres_i=node_i[2],node_i[3]
            old_feat_i=x_i[feat_index]
            new_feats.append(int(old_feat_i<thres_i) )
        return np.array(new_feats)

def make_tree_features(tree,threshold=4):
    node_depths=tree.tree_.compute_node_depths()
    selected_nodes= [ i for i,depth_i in enumerate(node_depths[1:])
                     if(depth_i<threshold)]
    selected_nodes.sort()
    tree_repr=tree.tree_.__getstate__()['nodes']
    return TreeFeatures(tree_repr,selected_nodes)