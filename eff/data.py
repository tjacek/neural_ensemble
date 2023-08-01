import numpy as np
from sklearn import preprocessing
from collections import Counter#defaultdict
from sklearn.model_selection import RepeatedStratifiedKFold

class Dataset(object):
    def __init__(self,X,y,cs=None):
        self.X=X
        self.y=y
        self.cs=cs 

    def get_splits(self,n_splits,n_repeats):
        cv = RepeatedStratifiedKFold(n_splits=n_splits, 
                                     n_repeats=n_repeats, 
                                     random_state=4)
        all_splits=[]
        for train_i,test_i in cv.split(self.X,self.y):
            all_splits.append((train_i,test_i))
        return all_splits

    def get_params(self):
        return {'n_cats':max(self.y)+1,
                'dims':self.X.shape[1],
                'batch':self.X.shape[0],
                'class_weights':dict(Counter(self.y))}

def prepare_data(df):
    X=df.iloc[:,:-1]
    X=X.to_numpy()
    X=preprocessing.scale(X)
    y=df.iloc[:,-1]
    cats={ cat_i:i for i,cat_i in enumerate(y.unique())}
    y=y.to_numpy()
    y=[cats[y_i] for y_i in y]
    return Dataset(X,np.array(y))