import os
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,BatchNormalization,Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
from sklearn.base import BaseEstimator, ClassifierMixin
from keras import callbacks
import time
import learn,nn,ovo

class NeuralEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self,hid_ratio=1,batch_ratio=0.5,l1=0.001,multi_clf='RF'):
        self.hid_ratio=hid_ratio
        self.l1=l1
        self.multi_clf=multi_clf
        self.batch_ratio=batch_ratio

    def fit(self,X,targets):
        n_cats=max(targets)+1
        y= np.array(binarize(targets))
        earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
        nn_params={'dims':X.shape[1],'n_cats':n_cats}
        n_hidd= int(self.hid_ratio*X.shape[1])
        l1=float(self.l1)
        ensemble=nn.BinaryEnsemble(n_hidd,l1)(nn_params)
        
        batch_size= int(self.batch_ratio * X.shape[0])
        start=time.time()
        ensemble.fit(X,y,epochs=100,
            batch_size=batch_size,verbose = 0)
#        print(f'Bulding model, {time.time()-start}')
        self.models,self.extractors=[],[]
        for cat_i in range(n_cats):
            extractor_i=Model(inputs=ensemble.input,
                outputs=ensemble.get_layer(f'hidden{cat_i}').output)
            self.extractors.append(extractor_i)           
            binary_i=extractor_i.predict(X)
            clf_i=learn.get_clf(self.multi_clf)#'LR')
            full_i=np.concatenate([X,binary_i],axis=1)
            clf_i.fit(full_i,targets)
            self.models.append(clf_i)

    def predict_proba(self,X):
        votes=[]
        for extr_i,model_i in zip(self.extractors,self.models):
            binary_i=extr_i.predict(X)
            full_i=np.concatenate([X,binary_i],axis=1)
            vote_i= model_i.predict_proba(full_i)
            votes.append(vote_i)
        votes=np.array(votes)
        prob=np.sum(votes,axis=0)
        return prob

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

def binarize(labels):
    n_cats=max(labels)+1
    y=[]
    for l_i in labels:
        vector_i=[]
        for j in range(n_cats):
            if(j==l_i):
                vector_i+=[1,0]
            else:
                vector_i+=[0,1]
        y.append(vector_i)
    return y

def get_ens(ens_type):
    if(ens_type=='one'):
        return ovo.OneVsOne
    return NeuralEnsemble