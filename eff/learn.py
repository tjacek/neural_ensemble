import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tools

class NECSCF(object):
    def __init__(self,dataset,split,cs_feats):
        self.dataset=dataset
        self.split=split
        self.cs_feats=cs_feats
        self.full_feats=None

    def __call__(self,clf_type):
        if(self.full_feats is None):
            common=self.dataset.X
            full=[[ np.concatenate([common,cs_j],axis=1) 
                        for cs_j in cs_i]
                            for cs_i in self.cs_feats]  
            full=[ self.split.get_dataset(X=full_i,
                                          y=self.dataset.y,
                                          i=i)
                        for i,full_i in enumerate(full)]
            self.full_feats=full 
        for full_i in self.full_feats:
            y_pred=ensemble(clf_type,full_i)
            test_exe= full_i[0][1]
            acc=test_exe.get_acc(y_pred)
            print(acc)
            
def ensemble(clf_type,datasets):
    votes=[]
    for train_i,test_i in datasets:
        clf_i=get_clf(clf_type)
        clf_i.fit(train_i.X,train_i.y)
        y_pred=clf_i.predict_proba(test_i.X)
        votes.append(y_pred)
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)

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