import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

class Dataset(object):
    def __init__(self,X,y=None):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape[1]
        
    def n_cats(self):
        return int(max(self.y))+1

    def fit_clf(self,train,clf):
        X_train,y_train=self.X[train],self.y[train]
        history=clf.fit(X_train,y_train)
        return clf,history

    def eval(self,train_index,test_index,clf,as_result=True):
        clf,history=self.fit_clf(train_index,clf)
        result=self.pred(test_index,clf)
        return result,history
    
    def pred(self,test_index,clf):
        if(test_index is None):
            X_test,y_test=self.X,self.y
        else:
            X_test,y_test=self.X[test_index],self.y[test_index]
        y_pred=clf.predict(X_test)
        return Result(y_pred,y_test)

    def params_dict(self):
        return {"n_cats":self.n_cats(),"dims":(self.dim(),)}

    def range(self):
        return [ (np.amin(x_i),np.amax(x_i))
                    for x_i in self.X.T]

    def weight_dict(self):
        cats=  list(set(self.y))
        n_cats= len(cats) 
        params={cat_i:0 for cat_i in cats}
        for y_i in self.y:
            params[y_i]+=1
        params={key_i:float(value_i) 
                   for key_i,value_i in params.items()}
        return params

class Result(object):
    def __init__(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true

    def get_acc(self):
        return accuracy_score(self.y_pred,self.y_true)

def read_csv(in_path:str):
    if(type(in_path)==tuple):
        X,y=in_path
        return Dataset(X,y)
    if(type(in_path)!=str):
        return in_path
    df=pd.read_csv(in_path,header=None)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    return Dataset(X,y)

if __name__ == '__main__':
    data=read_csv("wine-quality-red")
    print(data.range())