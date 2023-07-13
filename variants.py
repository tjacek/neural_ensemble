#from collections import defaultdict
import numpy as np
import learn#data,pred,tools

class NoEnsemble(object):
    def __init__(self,clfs):
        self.clfs=clfs

    def __call__(self,train,test):
        for clf_type_i in self.clfs:
            clf_i=learn.get_clf(clf_type_i)
            clf_i.fit(train.X,train.y)
            pred_i=clf_i.predict(test.X)
            yield clf_type_i,pred_i

class BasicVariant(object):
    def __init__(self,clfs):
        self.clfs=clfs

    def __call__(self,train,test):
        for clf_type_i in self.clfs:
            pred_i=necscf(train,test,clf_type_i)
            id_i=f'{clf_type_i}-necscf'
            yield id_i,pred_i

class BinaryVariant(object):
    def __init__(self,clfs):
        self.clfs=clfs

    def __call__(self,train,test):
        for cls_type_i in self.clfs:
            votes=[]
            for train_j,test_j in zip(train.cs,test.cs):
                clf_j=learn.get_clf(clf_type_i)
                clf_j.fit(train_j,train.y)
                y_pred=clf_j.predict_proba(test_j)
                votes.append(y_pred)
            votes=np.array(votes)
            votes=np.sum(votes,axis=0)
            y_pred=np.argmax(votes,axis=1)
            id_i=f'{clf_type_i}-binary'
            yield id_i,y_pred

def necscf(train,test,clf_type,votes=False):
    votes=[]
    for cs_train_i,cs_test_i in zip(train.cs,test.cs):
        full_train_i=np.concatenate([train.X,cs_train_i],axis=1)       
        clf_i=learn.get_clf(clf_type)
        clf_i.fit(full_train_i,train.y)
        full_test_i=np.concatenate([test.X,cs_test_i],axis=1)
        y_pred=clf_i.predict_proba(full_test_i)
        votes.append(y_pred)
    if(votes):
        return votes
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)