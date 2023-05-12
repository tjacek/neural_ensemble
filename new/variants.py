import numpy as np
from collections import namedtuple
import learn

class Ensemble(object):
    def __init__(self,train,test):
        self.train=train
        self.test=test
        self.clfs=[]

    def __call__(self,clf_type_i,variant_type_i='basic'):
        variant_i=get_variant(variant_type_i)
        clfs=variant_i(self,clf_type_i)
        votes=[]    
        common,binary=self.test.common,self.test.binary
        for j,clf_j in enumerate(clfs):
            multi_i=np.concatenate([common,binary[j]],axis=1)
            vote_i= clf_j.predict_proba(multi_i)
            votes.append(vote_i)
        votes=np.array(votes)
        prob=np.sum(votes,axis=0)
        return np.argmax(prob,axis=1)

def make_ensemble(nn_i,train_i,test_i):
    Dataset = namedtuple('Dataset','common binary targets')
    X_train,y_train=train_i
    binary_i=nn_i.binary_model.predict(X_train)
    train_data=Dataset(X_train,binary_i,y_train)
    X_test,y_test=test_i
    binary_i=nn_i.binary_model.predict(X_test)
    test_data=Dataset(X_test,binary_i,y_test)
    return Ensemble(train_data,test_data)

def get_variant(variant_type_i):
    if(variant_type_i=='basic'):
        return basic_variant

def basic_variant(inst_i,clf_type_i):
    clfs=[]
    common_i =inst_i.train.common
    for binary_j in inst_i.train.binary:
        multi_i=np.concatenate([common_i,binary_j],axis=1)
        clf_j =learn.get_clf(clf_type_i)
        clf_j.fit(multi_i,inst_i.train.targets)
        clfs.append(clf_j)
    return clfs