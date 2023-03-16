import numpy as np
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
import json
#from sklearn.utils import class_weight
import data

#class Votes(object):
#    def __init__(self,results):
#        self.results=results

#    def names(self):
#        return self.results[0].names()

#    def get_classes(self,name_i):
#        return [ np.argmax(result_i[name_i]) 
#                for result_i in self.results]

#    def dynamic_voting(self,s_dict):
#        result=Result()
#        for name_i,clfs_i in s_dict.items():
#            if(clfs_i is None):
#                ballot_i=[ r[name_i] 
#                    for r in self.results]
#            else:
#                ballot_i=[self.results[s][name_i] 
#                    for s in clfs_i]
#            count_i=np.sum(ballot_i ,axis=0)
#            result[name_i]=np.argmax(count_i)    
#        return result
    
#    def vote(self):
#        return voting(self.results)

class Result(data.DataDict):
    def get_pred(self):
        y_pred,y_true=[],[]
        for name_i,vote_i in self.items():
            y_pred.append(get_label(vote_i) )
            y_true.append(name_i.get_cat())
        return y_pred,y_true
    
    def types(self):
        return [type(key_i) for key_i in self]

    def get_acc(self):
        y_pred,y_true=self.get_pred()
        return accuracy_score(y_pred,y_true)
    
    def report(self):
        y_pred,y_true=self.get_pred()
        print(classification_report(y_true, y_pred,digits=4))
	
    def get_cf(self):
        y_pred,y_true=self.get_pred()
        return confusion_matrix(y_true, y_pred)

    def save(self,out_path):
        raw_dict={name_i:int(get_label(value_i))
            for name_i,value_i in self.items()}
        with open(out_path, 'wb') as f:
            json_str = json.dumps(raw_dict)         
            json_bytes = json_str.encode('utf-8') 
            f.write(json_bytes)

def get_label(vote_i):
    if(type(vote_i)==np.ndarray):
        return np.argmax(vote_i)
    else:
        return vote_i

def read_result(in_path):
    with open(in_path, 'r') as f:        
        json_str = f.read()                      
#        json_str = json_bytes.decode('utf-8')           
        raw_dict = json.loads(json_str)
        raw_dict= {data.Name(name_i):value_i
            for name_i,value_i in raw_dict.items()}
        return Result(raw_dict)

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
    if(type(clf_type)==str):
        clf_i=get_clf(clf_type)
    else:
        clf_i=clf_type
    clf_i.fit(X_train,y_train)
    X_test,y_true,names=test.as_dataset()
    y_pred=clf_i.predict_proba(X_test)
    return make_result(names,y_pred)

def get_clf(name_i):        
    if(name_i=="RF"):
        return ensemble.RandomForestClassifier()
    if(name_i=="LR(imb)"):
        return LogisticRegression(solver='liblinear',
            class_weight='balanced')
    if(name_i=='Bag'):
        return ensemble.BaggingClassifier()
    if(name_i=='Grad'):
        return ensemble.GradientBoostingClassifier()
    if(name_i=='MLP'):
        return MLPClassifier()
    return LogisticRegression(solver='liblinear')