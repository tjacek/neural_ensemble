import tensorflow as tf
from tensorflow.keras import Input, Model
import deep,utils

class Experiment(object):
    def __init__(self,split,hyper_params=None,model=None):
        self.split=split
        self.hyper_params=hyper_params
        self.model=model

    def train(self,alg_params,verbose=0):
        params=self.split.dataset.params
        x_train,y_train=self.split.get_train()
        y_train=[tf.keras.utils.to_categorical(y_train,
                                               num_classes=params['n_cats']) 
                    for k in range(params['n_cats'])]
        x_valid,y_valid=self.split.get_test()
        y_valid=[tf.keras.utils.to_categorical(y_valid,
                                               num_classes=params['n_cats']) 
                    for k in range(params['n_cats'])]
        if(verbose):
            self.model.summary()
        self.model.fit(x=x_train,
                       y=y_train,
                       batch_size=params['batch'],
                       epochs=alg_params.epochs,
                       validation_data=(x_valid, y_valid),
                       verbose=verbose,
                       callbacks=alg_params.get_callback())

    def eval(self,alg_params,clf_type="RF"):
        extractor=self.make_extractor()
        necscf=self.split.to_ncscf(extractor)
        necscf.train(clf_type=clf_type)
        return necscf.eval()

    def make_extractor(self):
        names= [ layer.name for layer in self.model.layers]
        n_cats=self.split.dataset.params['n_cats']
        penult=names[-2*n_cats:-n_cats]
        layers=[self.model.get_layer(name_i).output 
                    for name_i in penult]
        return Model(inputs=self.model.input,
                        outputs=layers)

    def save(self,out_path):
        utils.make_dir(out_path)
        self.model.save_weights(f'{out_path}/weights')
#        np.save(f'{out_path}/train',self.split.train_ind)
#        np.save(f'{out_path}/test',self.split.test_ind)
        with open(f'{out_path}/info',"a") as f:
            f.write(f'{str(self.hyper_params)}\n') 

def make_exp(split_i,hyper_params):
    model_i=deep.ensemble_builder(params=split_i.dataset.params,
                                  hyper_params=hyper_params,
                                  alpha=0.5)
    exp_i=Experiment(split=split_i,
                     hyper_params=hyper_params,
                     model=model_i)
    return exp_i 

def read_exp(in_path):
    with open(f'{in_path}/info',"r") as f:
        lines=f.readlines()
        print(lines)