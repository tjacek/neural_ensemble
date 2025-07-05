import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import dataset

class DataSplits(object):
    def __init__(self,data,splits):
        self.data=data
        self.splits=splits

class SplitProtocol(object):
    def __init__(self,n_splits,n_repeats):
        self.n_splits=n_splits
        self.n_repeats=n_repeats

    def get_split(self,data):
        rskf=RepeatedStratifiedKFold(n_repeats=self.n_repeats, 
                                     n_splits=self.n_splits, 
                                     random_state=0)
        splits=[]
        for train_index,test_index in rskf.split(data.X,data.y):
            splits.append(Split(train_index,test_index))
        return splits

class Split(object):
    def __init__(self,train_index,test_index):
        self.train_index=train_index
        self.test_index=test_index

        
    def eval(self,data,clf):
        return data.eval(train_index=self.train_index,
                         test_index=self.test_index,
                         clf=clf,
                         as_result=True)
       
    def fit_clf(self,data,clf):
        return data.fit_clf(self.train_index,clf)

    def pred(self,data,clf):
        return data.pred(self.test_index,
                         clf=clf,
                         as_result=True)

    def save(self,out_path):
        return np.savez(out_path,self.train_index,self.test_index)

    def __str__(self):
        train_size=self.train_index.shape[0]
        test_size=self.test_index.shape[0]
        return f"train:{train_size},test:{test_size}"

def get_clf(clf_type):
    if(clf_type=="RF"): 
        return RandomForestClassifier(class_weight="balanced")
    if(clf_type=="LR"):
        return LogisticRegression(solver='liblinear')
    if(clf_type=="SVM"):
        return svm.SVC(kernel='rbf')
    if(clf_type=="GRAD"):
        return GradientBoostingClassifier()
    raise Exception(f"Unknow clf type:{clf_type}")

def get_paths(out_path,ens_type,dirs):
    ens_path=f"{out_path}/{ens_type}"
    path_dir={dir_i:f"{ens_path}/{dir_i}" 
                    for dir_i in dirs}
    path_dir['ens']=ens_path
    path_dir['splits']=f"{out_path}/splits"
    return path_dir

def get_splits(data_path,
               n_splits=10,
               n_repeats=1):
    data=dataset.read_csv(data_path)
    protocol=SplitProtocol(n_splits,n_repeats)
    return DataSplits(data=data,
                      splits=protocol.get_split(data))