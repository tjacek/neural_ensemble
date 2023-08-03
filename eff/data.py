import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import Counter#defaultdict
from sklearn.model_selection import RepeatedStratifiedKFold

class Dataset(object):
    def __init__(self,X,y,cs=None):
        self.X=X
        self.y=y
        self.cs=cs 

    def get_splits(self,n_splits=10,n_repeats=3):
        cv = RepeatedStratifiedKFold(n_splits=n_splits, 
                                     n_repeats=n_repeats, 
                                     random_state=4)
        splits,current=[],[]
        for train_i,test_i in cv.split(self.X,self.y):
            current.append((train_i,test_i))
            if(len(current)==n_splits):
                splits.append(DataSplit(current))
                current=[]
        return splits

    def get_params(self):
        return {'n_cats':max(self.y)+1,
                'dims':self.X.shape[1],
                'batch':self.X.shape[0],
                'class_weights':dict(Counter(self.y))}

class DataSplit(object):
    def __init__(self,indices):
        self.indices=indices
    
    def check(self):
        test=[ test_i for train_i,test_i in self.indices]
        print(test)

#    def split_data(X,y):
#        splits=[ X[train_ind],y[train_ind]
#            for train_ind,t]

def get_dataset(data_path):
    df=pd.read_csv(data_path,header=None) 
    return prepare_data(df)

def prepare_data(df):
    X=df.iloc[:,:-1]
    X=X.to_numpy()
    X=preprocessing.scale(X)
    y=df.iloc[:,-1]
    cats={ cat_i:i for i,cat_i in enumerate(y.unique())}
    y=y.to_numpy()
    y=[cats[y_i] for y_i in y]
    return Dataset(X,np.array(y))