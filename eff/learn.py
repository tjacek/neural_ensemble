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
        
class AllFeatures(object):
    def __init__(self,features):
        self.features=features    

    def __call__(self,clfs,variants):
        for name_i,variant_i in variants.items():
            for clf_j in clfs:
                y=[(feat_k['y'].test,variant_i(clf_j,feat_k))
                    for feat_k in self.features]
                yield name_i,clf_j,y

def ensemble(clf_type,feat_i,feat_type='full'):
    votes=[]
    for j,full_j in enumerate(feat_i[feat_type]):
        clf_j=get_clf(clf_type)
        clf_j.fit(full_j.train,feat_i['y'].train)
        y_pred=clf_j.predict_proba(full_j.test)
        votes.append(y_pred)
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)

def common_variant(clf_type,feat_i):
    clf_j=get_clf(clf_type)
    clf_j.fit(feat_i['common'].train,feat_i['y'].train)
    return clf_j.predict(feat_i['common'].test)

def necscf_variant(clf_type,feat_i):
    return ensemble(clf_type,feat_i,feat_type='full')

def cs_variant(clf_type,feat_i):
    return ensemble(clf_type,feat_i,feat_type='cs')

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