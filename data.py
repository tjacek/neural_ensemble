import numpy as np
import pandas as pd
#import logging,time
from collections import namedtuple
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from collections import Counter#defaultdict
#import tensorflow as tf 

Dataset=namedtuple('Dataset','X y')
EnsDataset=namedtuple('EnsDataset','X y cs')

class AllSplits(object):
    def __init__(self,X,y,splits):
        self.X=X 
        self.y=y
        self.splits=splits

    def __call__(self):
        for split_i in self.splits: 
            train_data=split_i.get_train(self.X,self.y)
            test_data=split_i.get_test(self.X,self.y)
            train_ind=split_i.train_ind
            test_ind=split_i.test_ind
            yield (train_ind,train_data),(test_ind,test_data)
    
    def as_datasets(self):
        for split_i in self.splits:
            yield split_i.get_dataset(self.X,self.y)

class DataSplit(object):
    def __init__(self,train_ind,test_ind):
        self.train_ind=train_ind
        self.test_ind=test_ind

    def get_dataset(self,X,y):
        train= Dataset(*self.get_train(X,y))
        test= Dataset(*self.get_test(X,y))
        return train,test
    
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
    return AllSplits(X,y,all_splits)

def get_dataset(data_path):
    df=pd.read_csv(data_path,header=None) 
    return prepare_data(df)

def prepare_data(df):
    X=df.iloc[:,:-1]
    X=X.to_numpy()
    X=preprocessing.scale(X) # data preprocessing
    y=df.iloc[:,-1]
    cats={ cat_i:i for i,cat_i in enumerate(y.unique())}
    y=y.to_numpy()
    y=[cats[y_i] for y_i in y]
    return X,np.array(y)

def get_dataset_params(X,y):
    return {'n_cats':max(y)+1,
            'dims':X.shape[1],
            'batch':X.shape[0],
            'class_weights':dict(Counter(y))}
