import numpy as np
import pandas as pd
#import logging,time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing

class DataSplit(object):
    def __init__(self,train_ind,test_ind):
        self.train_ind=train_ind
        self.test_ind=test_ind

    def get_train(self,X,y):
        return X[self.train_ind],y[self.train_ind]

    def get_test(self,X,y):
        return X[self.test_ind],y[self.test_ind]

def gen_splits(X,y,n_splits=3,n_repeats=3):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, 
            n_repeats=n_repeats, random_state=4)
    all_splits=[]
    for train_i,test_i in cv.split(X,y):
        all_splits.append(DataSplit(train_i,test_i))
    return all_splits

def prepare_data(df):
    X=df.iloc[:,:-1]
    X=X.to_numpy()
    X=preprocessing.scale(X) # data preprocessing
    y=df.iloc[:,-1]
    cats={ cat_i:i for i,cat_i in enumerate(y.unique())}
    y=y.to_numpy()
    y=[cats[y_i] for y_i in y]
    return X,np.array(y)