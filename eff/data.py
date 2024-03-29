import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import Counter#defaultdict
from sklearn.model_selection import RepeatedStratifiedKFold
import tools

class Dataset(object):
    def __init__(self,X,y):#,cs=None):
        self.X=X
        self.y=y
#        self.cs=cs 

    def get_splits(self,n_splits=10,n_repeats=3):
        cv = RepeatedStratifiedKFold(n_splits=n_splits, 
                                     n_repeats=n_repeats, 
                                     random_state=4)
        splits,current=[],[]
        for train_i,test_i in cv.split(self.X,self.y):
            current.append((train_i,test_i))
            if(len(current)==n_splits):
                current= list(equalize(current))
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
    
    def __len__(self):
        return len(self.indices)
    
    def get_sizes(self):
        return [ train_i.shape[0]#,test_i.shape[0]) 
                for train_i,test_i in self.indices]

    def get_all(self,X,y=None,train=True):
        train=int(not train)
        if(y is None):
            splits=[ X[ind[train]]
                       for ind in self.indices]
            return splits
        else:
            splits=[ (X[ind[train]],y[ind[train]])
                    for ind in self.indices]
            X,y=list(zip(*splits))
            return X,y

    def get_dataset(self,X,y,i):
        train,test=self.indices[i]
        datasets=[]
        for x_i in X:
            X_train=x_i[train]
            y_train=y[train]
            X_test =x_i[test]
            y_test =y[test]
            datasets.append((Dataset(X_train,y_train),
                             Dataset(X_test,y_test)))
        return datasets
    
    def save(self,out_path):
        tools.make_dir(out_path)
        tools.make_dir(f'{out_path}/train')
        tools.make_dir(f'{out_path}/test')
        for i,(train_i,test_i) in enumerate(self.indices):
            np.save(f'{out_path}/train/{i}',train_i)
            np.save(f'{out_path}/test/{i}',test_i)

    def check(self):
        test=set()
        for train_i,test_i in self.indices:
            test.update(list(test_i))
        print(len(test))

def read_split(in_path):    
    train_ind=[np.load(path_i)
        for path_i in tools.top_files(f'{in_path}/train')]
    test_ind=[np.load(path_i)
        for path_i in tools.top_files(f'{in_path}/test')]
    indices=list(zip(train_ind,test_ind))
    return DataSplit(indices)

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

def equalize(single_split):
    sizes=[train_i.shape[0] 
        for train_i,test_i in single_split]
    final_size=max(sizes)
    for train_i,test_i in single_split:
        if(train_i.shape[0]<final_size):
            diff_i= final_size - train_i.shape[0]
            new_elem=[np.random.choice(train_i) 
                        for j in range(diff_i)]
            new_train= list(train_i)+new_elem
            yield np.array(new_train),test_i
        else:
            yield train_i,test_i