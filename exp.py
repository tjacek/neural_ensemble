import tensorflow as tf
from tensorflow.keras import Input, Model

class Experiment(object):
    def __init__(self,split,hyper_params=None,model=None):
        self.split=split
        self.hyper_params=hyper_params
        self.model=model

    def train(self,alg_params,verbose=0):
        params=self.split.dataset.params
        x_train,y_train=self.split.get_train()
        y_train=[tf.keras.utils.to_categorical(y_train) 
                    for k in range(params['n_cats'])]
        x_valid,y_valid=self.split.get_test()
        y_valid=[tf.keras.utils.to_categorical(y_valid) 
                    for k in range(params['n_cats'])]
        self.model.summary()
        raise Exception(y_train)
        self.model.fit(x=x_train,
                       y=y_train,
                       batch_size=params['batch'],
                       epochs=alg_params.epochs,
                       validation_data=(x_valid, y_valid),
                       verbose=verbose,
                       callbacks=alg_params.get_callback())

    def eval(self,alg_params):
        extractor=self.make_extractor()
        necscf=self.split.to_ncscf(extractor)
        necscf.train()
        return necscf.eval()
#        raise Exception(acc)

    def make_extractor(self):
        names= [ layer.name for layer in self.model.layers]
        n_cats=self.split.dataset.params['n_cats']
        penult=names[-2*n_cats:-n_cats]
        layers=[self.model.get_layer(name_i).output 
                    for name_i in penult]
        return Model(inputs=self.model.input,
                        outputs=layers)        