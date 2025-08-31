from sklearn.tree import DecisionTreeClassifier
import pandas as pd


class Dataset(object):
    def __init__(self,X,y,cols):
        self.X=X
        self.y=y
        self.cols=cols

    def __len__(self):
        return len(self.y)

def read_data(in_path,cols):
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    return Dataset(X,y,cols)

in_path="../multi_exp/data/car"
cols=[ 'buying','maint','doors', 
       'persons', 'lug_boot', 'safety' ]
data=read_data(in_path,cols)
print(len(data))