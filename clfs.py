import tensorflow as tf
import base

def get_clfs(clf_type):
    if(clf_type=="MLP"):
        return MLPFactory()
    raise Exception(f"Unknown clf type:{clf_type}")	

class MLPFactory(base.ClfFactory):
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

class MLP(base.ClfAdapter):

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