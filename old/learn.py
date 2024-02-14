import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.metrics import accuracy_score

def fit_clf(train,test,clf_type,hard=False,acc=False):
    clf_i=get_clf(clf_type)
    clf_i.fit(train.X,train.y)
    y_pred=clf_i.predict_proba(test.X)
    if(hard):
        y_pred=np.argmax(y_pred,axis=1)
    if(acc):
        return accuracy_score(y_pred,test.y)
    return (y_pred,test.y)

def get_clf(name_i):
    if(type(name_i)!=str):
        return name_i
    if(name_i=="SVC"):
        return SVC(probability=True)
    if(name_i=="RF"):
        return ensemble.RandomForestClassifier(class_weight='balanced_subsample')
    if(name_i=="LR"):
        return LogisticRegression(solver='liblinear',
            class_weight='balanced')