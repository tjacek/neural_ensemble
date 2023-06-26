import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import ensemble

def fit_clf(X,y,split_i,clf_type=None,hard=False):#,balance=False):
    clf_i= get_clf(clf_type)
    train_X,train_y=split_i.get_train(X,y)
    clf_i.fit(train_X,train_y)
    test_X,test_y=split_i.get_train(X,y)
    y_pred=clf_i.predict_proba(test_X)
    if(hard):
        y_pred=np.argmax(y_pred,axis=1)
    return (y_pred,test_y)

def get_clf(name_i):
    if(type(name_i)!=str):
        return name_i
    if(name_i=="SVC"):
        return SVC(probability=True)
    if(name_i=="RF"):
        return ensemble.RandomForestClassifier(class_weight='balanced_subsample')
    if(name_i=="LR-imb"):
        return LogisticRegression(solver='liblinear',
            class_weight='balanced')