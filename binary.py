import os
import numpy as np
import tensorflow as tf
import tensorflow.keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,BatchNormalization,Concatenate
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
from sklearn.base import BaseEstimator, ClassifierMixin
from keras import callbacks
import time
import learn,nn

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

class OneVsOne(NeuralEnsemble):
    def fit(self,X,y):
        pairs=self.select_pairs(X,y)
        self.extractors,self.models=[],[]
        for i,j in pairs:
            selected=[k for k,y_k in enumerate(y)
                        if((y_k==i) or (y_k==j))]
            y_s=[ int(y[k]==i) for k in selected]                
            y_s= tf.keras.utils.to_categorical(y_s, num_classes = 2)
            X_s=np.array([ X[k,:] for k in selected])
            extractor_i=self.train_extractor(X_s,y_s)
            binary_i=extractor_i.predict(X)
            self.train_model(X,binary_i,y)

    def select_pairs(self,X,y):
        train,test=split_dataset(X,np.array(y))
        cf=self.make_cf(train,test)
        n_cats= max(y)+1
        cat_size={ cat_i: y.count(cat_i) 
            for cat_i in range(n_cats)}
        pairs,values=[],[]
        for i in range(n_cats):
            for j in range(n_cats):
                if(i!=j):
                    size_ij=min(cat_size[i],cat_size[j])
                    cf_ij=cf[i][j]/float(size_ij)
                    values.append(cf_ij)
                    pairs.append((i,j))
        pairs=[ pairs[i]
            for i in np.argsort(values)[-n_cats:]]
        return pairs

    def make_cf(self,train,test):
        clf=learn.get_clf(self.multi_clf)
        clf.fit(train[0],train[1])
        y_pred=clf.predict(test[0])
        return confusion_matrix(y_pred,test[1])


    def train_extractor(self,X,y_i):
        earlystopping = callbacks.EarlyStopping(monitor="accuracy",
                mode="min", patience=5,restore_best_weights=True)
        nn_params={'dims':X.shape[1],'n_cats':2}
        n_hidd= int(self.hid_ratio*X.shape[1])
        l1=float(self.l1)
        start=time.time()
        nn_i=nn.SimpleNN(n_hidden=n_hidd,l1=l1)(nn_params)
        batch_size= int(self.batch_ratio * X.shape[0])
        start=time.time()
        nn_i.fit(X,y_i,epochs=100,
            batch_size=batch_size,verbose = 0)#,callbacks=earlystopping)
        extractor_i= nn.get_extractor(nn_i)
        self.extractors.append(extractor_i)
        return extractor_i

    def train_model(self,X,binary_i,targets):
        clf_i=learn.get_clf(self.multi_clf)#'LR')
        full_i=np.concatenate([X,binary_i],axis=1)
        clf_i.fit(full_i,targets)
        self.models.append(clf_i)

def split_dataset(X,y):
    skf = StratifiedKFold(n_splits=2)
    splits=[ split_i
        for split_i in skf.split(X, y)]
    train,test=splits
    train_X,train_y=X[train[0]],y[train[0]]
    test_X,test_y=X[test[0]],y[test[0]]
    return (train_X,train_y),(test_X,test_y)

def get_ens(ens_type):
    if(ens_type=='one'):
        return OneVsOne
    return NeuralEnsemble