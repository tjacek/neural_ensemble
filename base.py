import tensorflow as tf
import numpy as np
import json,random
import data

class AlgParams(object):
    def __init__(self,hyper_type='eff',epochs=300,callbacks=None,alpha=None,
                    bayes_iter=5,rest_clf=None):
        if(alpha is None):
            alpha=[0.25,0.5,0.75]
        self.hyper_type=hyper_type
        self.epochs=epochs
        self.alpha=alpha
        self.bayes_iter=bayes_iter
        self.rest_clf=rest_clf

    def optim_alpha(self):
        return type(self.alpha)==list 

    def get_callback(self):
        return tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=5)

class Protocol(object):
    def __init__(self,n_split=10,n_iters=10):
        self.n_split=n_split
        self.n_iters=n_iters
        self.current_split=None

    def set_split(self,dataset):
        train,test=self.gen_split(dataset)
        self.current_split=Split(dataset=dataset,
                                 train=train,
                                 test=test)

    def iter(self,dataset):
        for i in range(self.n_iters):
            self.set_split(dataset)
            yield self.current_split

    def gen_split(self,dataset):
        by_cat=dataset.by_cat()
        train,test=[],[]
        for cat_i,samples_i in by_cat.items():
            random.shuffle(samples_i)
            for j,index in enumerate(samples_i):
                if( (j% self.n_split)==0):
                    test.append(index)
                else:
                    train.append(index)
        return train,test

class Split(object):
    def __init__(self,dataset,train,test):
        self.dataset=dataset
        self.train=train
#        self.valid=valid
        self.test=test
    
    def get_train(self):
        return self.dataset.X[self.train],self.dataset.y[self.train]
    
    def get_valid(self):
        return self.get_train()#self.dataset.X[self.valid],self.dataset.y[self.valid]

    def get_test(self):
        return self.dataset.X[self.test],self.dataset.y[self.test]

    def to_ncscf(self,extractor):
        all_splits=[]
        for cs_i in extractor.predict(self.dataset.X):
            feats_i=np.concatenate([self.dataset.X,cs_i],axis=1)
            data_i=data.Dataset(X=feats_i,
                                y=self.dataset.y,
                                params=self.dataset.params)
            split_i=Split(dataset=data_i,
                          train=self.train,
                          test=self.test)
            all_splits.append(split_i)
        return NECSCF(all_splits=all_splits)


class NECSCF(object):
    def __init__(self,all_splits):
        self.all_splits=all_splits
        self.clf=[]