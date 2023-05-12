import numpy as np
import sklearn.metrics
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn import ensemble
import json
#from sklearn.utils import class_weight

class OptimClf(BaseEstimator, ClassifierMixin):
    def __init__(self,clf_name,hyper):
        self.hyper=hyper
        self.clf_name=clf_name
        self.clf_class=get_clf(self.clf_name)
        self.model=None
        if(self.clf_name=='SVC'):
            self.hyper['probability']=True
    
    def __call__(self):
        return self

    def fit(self,X,targets):
#        self.clf_class=
        if(self.hyper is None):
            self.model=self.clf_class()
        else:
            self.model= self.clf_class(**self.hyper)
        self.model.fit(X,targets)
        return self

    def predict_proba(self,X):
        return self.model.predict_proba(X)

    def predict(self,X):
        return self.model.predict(X)

    def __str__(self):
        return f'Optimised {self.clf_name}'

def get_search_space(clf_type):
    if(clf_type=='RF'):
        return {
            'n_estimators': [10, 25, 50, 100, 200], 
            'criterion': ['gini', 'entropy'], 
            'max_depth': [3, 5, 10, 15, 20, 25, None], 
            'min_samples_leaf': [1, 2, 5, 10]
        }
    return {'C': [0.1,1, 10, 100], 
            'gamma': [1,0.1,0.01,0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']}

def get_metric(metric_type):
    if(metric_type=='acc'):
        return accuracy_score
#    if(metric_type=='balanced_acc'):
#        return balance
    if(metric_type=='f1'):
        return f1_metric
    if(metric_type=='recall'):
        return recall_metric
    if(metric_type=='precision'):
        return precision_metric

def f1_metric(y_pred,y_true):
    return sklearn.metrics.f1_score(y_pred,y_true,average='macro')

def recall_metric(y_pred,y_true):
    return sklearn.metrics.recall_score(y_pred,y_true,average='macro')

def precision_metric(y_pred,y_true):
    return sklearn.metrics.precision_score(y_pred,y_true,average='macro')

def get_clf(name_i):
    if(type(name_i)!=str):
        return name_i
    if(name_i=="RF"):
        return ensemble.RandomForestClassifier(class_weight='balanced_subsample')
    if(name_i=="LR-imb"):
        return LogisticRegression(solver='liblinear',
            class_weight='balanced')
    if(name_i=='Bag'):
        return ensemble.BaggingClassifier()
    if(name_i=='Grad'):
        return ensemble.GradientBoostingClassifier#()
    if(name_i=='MLP'):
        return MLPClassifier#()
    if(name_i=="SVC"):
        return SVC(probability=True)
    return LogisticRegression#(solver='liblinear')