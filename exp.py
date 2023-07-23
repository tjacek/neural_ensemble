import tools
tools.silence_warnings()
import numpy as np
import tensorflow as tf 
from keras import callbacks

class ExpFactory(object):
    def __init__(self,dataset_params,hyper_params):
        self.dataset_params=dataset_params
        self.hyper_params=hyper_params	
        self.early_stop = callbacks.EarlyStopping(monitor='accuracy',
                                                  mode="max", 
                                                  patience=5,
                                                  restore_best_weights=True)

    def __call__(self,X,y,split,ens_type):
        make_model=deep.get_ensemble(ens_type)
        model=make_model(self.dataset_params,self.hyper_params)
        X_train,y_train=split.get_train(X,y)
        y_train = tf.keras.utils.to_categorical(y_train, 
        	                                    num_classes = self.dataset_params['n_cats'])
        history=model.fit(X_train,y_train,
        	              batch_size=self.dataset_params['batch'],
                          epochs=150,
                          verbose=0,
                          callbacks=self.early_stop)
        return Exp(split,
        	       model,
        	       ens_type,
        	       self.hyper_params,
        	       self.dataset_params)

class Exp(object):
    def __init__(self,split,model,ens_type,hyper,params):
        self.split=split
        self.model=model
        self.ens_type=ens_type
        self.hyper=hyper
        self.params=params

    def is_ens(self):
    	return isinstance(self.model,deep.NeuralEnsemble) 
    
    def get_features(self,X,y):
        train,test=self.split.get_dataset(X,y)
        if(self.is_ens()):
            cs_train=self.model.extract(train.X)
            cs_test=self.model.extract(test.X)
            train=data.EnsDataset(train.X,train.y,cs_train)
            test=data.EnsDataset(test.X,test.y,cs_test)
        return train,test

    def save(self,out_path):
        self.model.save_weights(f'{out_path}/weights')
        np.save(f'{out_path}/train',self.split.train_ind)
        np.save(f'{out_path}/test',self.split.test_ind)
        with open(f'{out_path}/info',"a") as f:
            f.write(f'{self.ens_type}\n')
            f.write(f'{str(self.hyper)}\n') 
            f.write(f'{str(self.params)}\n') 

def read_exp(in_path):
    train_ind=np.load(f'{in_path}/train.npy')
    test_ind=np.load(f'{in_path}/test.npy')
    split=data.DataSplit(train_ind,test_ind)
    with open(f'{in_path}/info',"r") as f:
        lines=f.readlines()
        ens_type= lines[0].strip()
        if(tools.has_number(ens_type)):
            ens_type=eval(ens_type)	
        make_model=deep.get_ensemble(ens_type)
        hyper,params= eval(lines[1]),eval(lines[2])
        model=make_model(params,hyper)
        model.load_weights(f'{in_path}/weights')
        return Exp(split,model,ens_type,hyper,params)