import numpy as np
from collections import namedtuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import learn,inliner

class Ensemble(object):
    def __init__(self,train,test):
        self.train=train
        self.test=test

    def __len__(self):
        return len(self.train.binary)
    
    def n_cats(self):
        return max(self.train.targets)+1

    def __call__(self,clf_type_i,variant_type_i='basic'):
        variant_i=get_variant(variant_type_i)
        return variant_i(self,clf_type_i)

    def get_true(self):
        return self.test.targets

    def get_acc(self,y_pred):
        if( len(y_pred.shape)==2):
            y_pred=np.argmax( y_pred,axis=1)
        return accuracy_score(self.get_true(),y_pred)

def make_ensemble(nn_i,train_i,test_i,s_clf=None):
    Dataset = namedtuple('Dataset','common binary targets')
    X_train,y_train=train_i
    binary_i=nn_i.binary_model.predict(X_train)
    if(not (s_clf is None)):
        binary_i=[ binary_i[k]  for k in s_clf ]
    train_data=Dataset(X_train,binary_i,y_train)
    X_test,y_test=test_i
    binary_i=nn_i.binary_model.predict(X_test)
    if(not (s_clf is None)):
        binary_i=[ binary_i[k]  for k in s_clf ]
    test_data=Dataset(X_test,binary_i,y_test)
    return Ensemble(train_data,test_data)

def get_variant(variant_type_i):
    if(variant_type_i=='NECSCF'):
        return BasicVariant(False)
    if(variant_type_i=='NECSCF2'):
        return BasicVariant(True)
    if(variant_type_i=='common'):
        return common_variant
    if(variant_type_i=='binary'):
        return binary_variant
    if(variant_type_i=='better'):
        return better_variant
    if(variant_type_i=='best'):
        return best_variant
    if(variant_type_i=='inliner'):
        return inliner.InlinerVoting()
    if(variant_type_i=='conf'):
        return inliner.conf_voting
    raise Exception(f'Variant {variant_type_i} not implemented')

class BasicVariant(object):
    def __init__(self,common=False):
        self.common=False

    def __call__(self,inst_i,clf_type_i):
        if(len(inst_i)==0):
            return common_variant(inst_i,clf_type_i)  
        clfs=train_clfs(clf_type_i,inst_i.train)
        votes=eval_clfs(clfs,inst_i.test)
        if(self.common):
            raw=common_variant(inst_i,clf_type_i)
            votes.append(to_one_hot(raw,inst_i))
        return voting(votes)

def binary_variant(inst_i,clf_type_i):
    votes=[]
    for train_j,test_j in zip(inst_i.train.binary,inst_i.test.binary):
        clf_j =learn.get_clf(clf_type_i)
        clf_j.fit(train_j,inst_i.train.targets)
        vote_j= clf_j.predict_proba(test_j)
        votes.append(vote_j)
    return voting( votes)

def common_variant(inst_i,clf_type_i):
    clf_j =learn.get_clf(clf_type_i)
    clf_j.fit(inst_i.train.common,inst_i.train.targets)
    return clf_j.predict(inst_i.test.common)

def best_variant(inst_i,clf_type_i):
    clfs=train_clfs(clf_type_i,inst_i.train)
    votes=eval_clfs(clfs,inst_i.test)
    indiv_acc=[]
    for vote_i in votes:
        acc_i=inst_i.get_acc(vote_i)
        indiv_acc.append(acc_i)
    k=np.argmax(indiv_acc)
    return np.argmax(votes[k],axis=1) #voting(s_votes)

def better_variant(inst_i,clf_type_i):
    y_pred=common_variant(inst_i,clf_type_i)
    common_acc=inst_i.get_acc(y_pred)
    clfs=train_clfs(clf_type_i,inst_i.train)
    votes=eval_clfs(clfs,inst_i.test)
    s_votes=[]
    for vote_i in votes:
        acc_i=inst_i.get_acc(vote_i)
        if(acc_i>common_acc):
            s_votes.append(vote_i)
    if(len(s_votes)==0):
        return y_pred#np.argmax(y_pred,axis=1)
    return voting(s_votes)

def train_clfs(clf_type_i,dataset):
    clfs=[]
    for binary_j in dataset.binary:
        multi_i=np.concatenate([dataset.common,binary_j],axis=1)
        clf_j =learn.get_clf(clf_type_i)
        clf_j.fit(multi_i,dataset.targets)
        clfs.append(clf_j)
    return clfs

def eval_clfs(clfs,dataset):
    votes=[]
    for j,clf_j in enumerate(clfs):
        binary_j=dataset.binary[j]
        multi_i=np.concatenate([dataset.common,binary_j],axis=1)
        vote_i= clf_j.predict_proba(multi_i)
        votes.append(vote_i)
    return votes

def voting(votes):
    votes=np.array(votes)
    prob=np.sum(votes,axis=0)
    return np.argmax(prob,axis=1)

def to_one_hot(raw,inst_i):
    if(len(raw.shape)==1):
        base_vote = np.zeros((raw.size, inst_i.n_cats()))
        base_vote[np.arange(raw.size), raw] = 1
    else:
        base_vote=raw
    return base_vote