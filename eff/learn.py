import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from collections import namedtuple
import tools

Features=namedtuple('Features','train test')

class FeaturesFactory(object):
    def __init__(self):
        self.feat_types={
          'common':common_feats,
          'cs':cs_feats,
          'full':full_feats
        }
    
    def __call__(self,dataset,split,cs_feats):
        common,y=dataset.X,dataset.y
        features=[]
        for (train,test),cs_i in zip(split.indices,cs_feats):
            def helper(x):
                if(type(x)==list):
                    return [helper(x_i) for x_i in x]
                x_train=x[train]
                x_test=x[test]
                return Features(x_train,x_test)
            feats_i={ name_j: helper(type_j(common,cs_i))
                      for name_j,type_j in self.feat_types.items()}
            feats_i['y']=helper(y)
            features.append(feats_i)
        return AllFeatures(features)

def common_feats(common,cs):
    return common

def cs_feats(common,cs):
    return cs

def full_feats(common,cs):
    return [np.concatenate([common,cs_j],axis=1)
                for cs_j in cs]

#class CSFeatures(object):
#    def __init__(self,common,cs,full,y):
#        self.common_train=common
#        self.cs=cs
#        self.full=full
#        self.y=y
        
class AllFeatures(object):
    def __init__(self,features):
        self.features=features    

    def __call__(self,clf_type):
        acc=tools.get_metric('acc')
        for feat_i in self.features:
            y_pred=ensemble(clf_type,feat_i)
            acc_i=acc(feat_i['y'].test,y_pred)
            print(acc_i)

def ensemble(clf_type,feat_i):
    votes=[]
    for j,full_j in enumerate(feat_i['full']):
        clf_j=get_clf(clf_type)
        clf_j.fit(full_j.train,feat_i['y'].train)
        y_pred=clf_j.predict_proba(full_j.test)
        votes.append(y_pred)
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)

def make_features(dataset,split,cs_feats):
    factory= FeaturesFactory()
    return factory(dataset,split,cs_feats)

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