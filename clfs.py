import tensorflow as tf
import base

def get_clfs(clf_type):
    if(clf_type in base.OTHER_CLFS):
        return base.ClasicalClfFactory(clf_type)
    if(clf_type=="MLP"):
        return MLPFactory()
    raise Exception(f"Unknown clf type:{clf_type}")	

class ClfFactory(object):
    def __init__(self,hyper_params=None,
                      loss_gen=None):
        if(hyper_params is None):
           hyper_params=default_hyperparams()
        self.params=None
        self.hyper_params=hyper_params
        self.class_dict=None
        self.loss_gen=loss_gen
    
    def init(self,data):
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'n_epochs':1000}
        self.class_dict=dataset.get_class_weights(data.y)

    def __call__(self):
        raise NotImplementedError()

    def read(self,model_path):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

class ClfAdapter(object):
    def __init__(self, params,
                       hyper_params,
                       class_dict=None,
                       model=None,
                       loss_gen=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.class_dict=class_dict
        self.model = model
        self.loss_gen=loss_gen
        self.verbose=verbose

    def fit(self,X,y):
        raise NotImplementedError()

    def eval(self,data,split_i):
        test_data_i=data.selection(split_i.test_index)
        raw_partial_i=self.partial_predict(test_data_i.X)
        result_i=dataset.PartialResults(y_true=test_data_i.y,
                                        y_partial=raw_partial_i)
        return result_i

    def save(self,out_path):
        raise NotImplementedError()



class MLPFactory(object):
    def __call__(self):
        return Deep(params=self.params,
                    hyper_params=self.hyper_params,
                    class_dict=self.class_dict)
    
    def read(self,model_path):
        model_i=tf.keras.models.load_model(model_path,
                                           custom_objects={"loss":deep.WeightedLoss})
        clf_i=self()
        clf_i.model=model_i
        return clf_i

    def get_info(self):
        return {"ens":"deep","callback":"total","hyper":self.hyper_params}

class MLP(object):

    def fit(self,X,y):
        if(self.model is None):
            self.model=deep.single_builder(params=self.params,
                                           hyper_params=self.hyper_params,
                                           class_dict=self.class_dict)
        y=tf.one_hot(y,depth=self.params['n_cats'])
        return self.model.fit(x=X,
                              y=y,
                              epochs=self.params['n_epochs'],
                              callbacks=ens_depen.basic_callback(),
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

def basic_callback():
    return tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                            patience=15)