import numpy as np
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
#from sklearn.utils import class_weight
import data

class Votes(object):
    def __init__(self,results):
        self.results=results

    def names(self):
        return self.results[0].names()

    def get_classes(self,name_i):
        return [ np.argmax(result_i[name_i]) 
                for result_i in self.results]

    def dynamic_voting(self,s_dict):
        result=Result()
        for name_i,clfs_i in s_dict.items():
            if(clfs_i is None):
                ballot_i=[ r[name_i] 
                    for r in self.results]
            else:
                ballot_i=[self.results[s][name_i] 
                    for s in clfs_i]
            count_i=np.sum(ballot_i ,axis=0)
            result[name_i]=np.argmax(count_i)    
        return result
    
    def vote(self):
        return voting(self.results)

class Result(data.DataDict):
    def get_pred(self):
        y_pred,y_true=[],[]
        for name_i,vote_i in self.items():
            if(type(vote_i)==np.ndarray):
                y_pred.append(np.argmax(vote_i))
            else:
                y_pred.append(vote_i)
            y_true.append(name_i.get_cat())
        return y_pred,y_true
    
    def get_acc(self):
        y_pred,y_true=self.get_pred()
        return accuracy_score(y_pred,y_true)

    def report(self):
        y_pred,y_true=self.get_pred()
        print(classification_report(y_true, y_pred,digits=4))
	
def make_result(names,y_pred):
    result=[(name_i,pred_i) 
            for name_i,pred_i in zip(names,y_pred)]
    return Result(result)

def unify_results(results):
    pairs=[]
    for result_i in results:
        pairs+=result_i.items()
    return Result(pairs)

def voting(results):
    names= results[0].keys()
    pairs=[]
    for name_i in names:
        ballot_i=[result_i[name_i] 
            for result_i in results]
        count_i=np.sum(ballot_i ,axis=0)
        cat_i=np.argmax(count_i)
        pairs.append((name_i,cat_i))
    return Result(pairs)

def fit_clf(data_dict_i,clf_type=None,balance=False):
    data_dict_i.norm()
    train,test= data_dict_i.split()
    X_train,y_train,names=train.as_dataset()

#    if(clf_i is None):
#        if(balance):
#            clf_i=LogisticRegression(solver='liblinear',
#                class_weight='balanced')
#        else:
#            clf_i=LogisticRegression(solver='liblinear')
    clf_i=get_clf(clf_type)
    clf_i.fit(X_train,y_train)
    X_test,y_true,names=test.as_dataset()
    y_pred=clf_i.predict_proba(X_test)
    return make_result(names,y_pred)

def get_clf(name_i):        
    if(name_i=="RF"):
        return ensemble.RandomForestClassifier()
    if(name_i=='Bag'):
        return ensemble.BaggingClassifier()
    if(name_i=='Grad'):
        return ensemble.GradientBoostingClassifier()
    return LogisticRegression(solver='liblinear')