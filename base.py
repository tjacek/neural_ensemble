import tensorflow as tf
import numpy as np
import json#,random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score,classification_report,f1_score
from sklearn import ensemble
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

    def single_split(self,dataset):
        raise NotImplementedError()

    def iter(self,dataset):
        raise NotImplementedError()

    def add_exp(self,exp_i):
        raise NotImplementedError()  

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

    def eval(self,clf_type):
        clf_i=get_clf(clf_type)
        X_train,y_train=self.get_train()
        clf_i.fit(X_train,y_train)
        X_test,y_test=self.get_test()
        y_pred=clf_i.predict(X_test)
        return Result(y_true=y_test,
                      y_pred=y_pred)

    def __str__(self):
        return f"{len(self.train)}:{len(self.test)}"

class NECSCF(object):
    def __init__(self,all_splits):
        self.all_splits=all_splits
        self.clfs=[]

    def train(self,clf_type="RF"):
        for split_i in self.all_splits:
            clf_i=get_clf(clf_type)
            X_train,y_train=split_i.get_train()
            clf_i.fit(X_train,y_train)
            self.clfs.append(clf_i)

    def eval(self):
        votes=[]
        for split_i,clf_i in zip(self.all_splits,self.clfs):
            X_test,y_test=split_i.get_test()
            votes.append(clf_i.predict_proba(X_test).T)
        votes= np.array(votes)
        y_pred= np.sum(votes,axis=0).T
        y_pred=np.argmax(y_pred,axis=1)
        return Result(y_true=y_test,
                      y_pred=y_pred)

class Result(object):
    def __init__(self,y_true,y_pred):
        self.y_true=y_true
        self.y_pred=y_pred

    def acc(self):
        return accuracy_score(self.y_true,self.y_pred)

def get_clf(name_i):
    if(type(name_i)!=str):
        return name_i
    if(name_i=="RF"):
        return ensemble.RandomForestClassifier(class_weight='balanced_subsample')
    if(name_i=="LR-imb"):
        return LogisticRegression(solver='liblinear',
            class_weight='balanced')