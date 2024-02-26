import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from collections import Counter
from collections import defaultdict
import utils

class Dataset(object):
    def __init__(self,X,y,params):
        self.X=X
        self.y=y
        self.params=params

    def by_cat(self):
        by_cat=defaultdict(lambda :[])
        for i,cat_i in enumerate(self.y):
            by_cat[cat_i].append(i)
        return by_cat

def get_data(in_path):
    df=pd.read_csv(in_path)
    X,y=prepare_data(df,target=-1)
    params=get_dataset_params(X,y)
    return Dataset(X=X,
                   y=y,
                   params=params)

def prepare_data(df,target=-1):
    to_numeric(df)
    X=df.to_numpy()
    X=np.delete(X,[target], axis=1)
    X=np.nan_to_num(X)
#    X=preprocessing.scale(X) # data preprocessing
#    print(f'X:{np.mean(X)}')
    X= preprocessing.RobustScaler().fit_transform(X)
    y=df.iloc[:,target]
    cats={ cat_i:i for i,cat_i in enumerate(y.unique())}
    y=y.to_numpy()
    y=[cats[y_i] for y_i in y]
    return X,np.array(y)

def to_numeric(df):
    for col_i,type_i in zip(df.columns,df.dtypes):
        if(type_i=='object'):
            values={value_i:i 
                 for i,value_i in enumerate(df[col_i].unique())}
            df[col_i]=df[col_i].apply(lambda x: values[x])
    return df

def get_dataset_params(X,y):
    return {'n_cats':max(y)+1,
            'dims':X.shape[1],
            'batch':X.shape[0],
            'class_weights':dict(Counter(y))}