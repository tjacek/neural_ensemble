import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from collections import namedtuple
import tools

Features=namedtuple('Features','train test')

class CSFeatures(object):
    def __init__(self,common,cs,full,y):
        self.common_train=common
        self.cs=cs
        self.full=full
        self.y=y
        
class AllFeatures(object):
    def __init__(self,features):
        self.features=features    

    def __call__(self,clf_type):
        acc=tools.get_metric('acc')
        for feat_i in self.features:
            y_pred=ensemble(clf_type,feat_i)
            acc_i=acc(feat_i.test.y,y_pred)
            print(acc_i)

def ensemble(clf_type,feat_i):
    votes=[]
    for j,full_j in enumerate(feat_i.train.full):
        clf_j=get_clf(clf_type)
        clf_j.fit(full_j,feat_i.train.y)
        y_pred=clf_j.predict_proba(feat_i.test.full[j])
        votes.append(y_pred)
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)

def make_features(dataset,split,cs_feats):
    features=[]
    common,y=dataset.X,dataset.y
    for (train,test),cs_i in zip(split.indices,cs_feats):
        full=[np.concatenate([common,cs_j],axis=1)
                for cs_j in cs_i]
        common_train=common[train]
        common_test=common[test]
        y_train=y[train]
        y_test=y[test]
        full_train,full_test=zip(*[ (full_j[train],
                                     full_j[test]) 
                                        for full_j in full])
        cs_train,cs_test=zip(*[ (cs_j[train],
                                 cs_j[test]) 
                                    for cs_j in cs_i])
        train_feats=CSFeatures(common=common_train,
                               cs=cs_train,
                               full=full_train,
                               y=y_train)
        test_feats=CSFeatures(common=common_test,
                              cs=cs_test,
                              full=full_test,
                              y=y_test)
        features.append(Features(train=train_feats,
                                 test=test_feats))
    return AllFeatures(features)

def get_clf(name_i):
    if(type(name_i)!=str):
        return name_i
    if(name_i=="SVC"):
        return SVC(probability=True)
    if(name_i=="RF"):
        return RandomForestClassifier(class_weight='balanced_subsample')
    if(name_i=="LR"):
        return LogisticRegression(solver='liblinear',
                                  class_weight='balanced')