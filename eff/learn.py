import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class NECSCF(object):
    def __init__(self,dataset,split,cs_feats):
        self.dataset=dataset
        self.split=split
        self.cs_feats=cs_feats
        self.full_feats=None

    def __call__(self,clf_type):
        if(self.full_feats is None):
            common=self.dataset.X
            self.full_feats=[[ np.concatenate([common,cs_j],axis=1) 
                                for cs_j in cs_i]
                                    for cs_i in self.cs_feats]	
        for full_i in self.full_feats:
            print(full_i[0].shape)	

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