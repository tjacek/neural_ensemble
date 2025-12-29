import numpy as np
import os.path
from abc import ABC, abstractmethod
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm
import dataset,utils

class Split(object):
    def __init__(self,train_index,test_index):
        self.train_index=train_index
        self.test_index=test_index

    def eval(self,data,clf):
        if(type(clf)==str):
            clf=get_clf(clf)
        return data.eval(train_index=self.train_index,
                         test_index=self.test_index,
                         clf=clf)
       
    def fit_clf(self,data,clf):
        return data.fit_clf(self.train_index,clf)

    def pred(self,data,clf):
        return data.pred(self.test_index,
                         clf=clf)

    def pred_partial(self,data,clf):
        return data.pred_partial(self.test_index,
                                 clf=clf)

    def save(self,out_path):
        return np.savez(out_path,self.train_index,self.test_index)

    def __str__(self):
        train_size=self.train_index.shape[0]
        test_size=self.test_index.shape[0]
        return f"train:{train_size},test:{test_size}"

def read_split(in_path):
    raw_split=np.load(in_path)
    return Split(train_index=raw_split["arr_0"],
                 test_index=raw_split["arr_1"])

def read_split_dir(in_path):
    return [read_split(path_i)  
               for path_i in utils.top_files(in_path)]
    
def random_split(n_samples,p=0.9):
    if(type(n_samples)==dataset.Dataset):
        n_samples=len(n_samples)
    train_index,test_index=[],[]
    for i in range(n_samples):
        if(np.random.rand()<p):
            train_index.append(i)
        else:
            test_index.append(i)
    return Split(train_index=np.array(train_index),
                 test_index=np.array(test_index))

def get_splits( data,
                n_splits=10,
                n_repeats=1):
    if(type(data)==str):
        data=dataset.read_csv(data)
    rskf=RepeatedStratifiedKFold(n_repeats=n_repeats, 
                                 n_splits=n_splits, 
                                 random_state=0)
    splits=[]
    for train_index,test_index in rskf.split(data.X,data.y):
        splits.append(Split(train_index,test_index))
    return splits


def make_split_dir( in_path,
                    out_path,
                    n_splits=10,
                    n_repeats=3):
    split_path=f"{out_path}/splits"
    if(os.path.exists(split_path)):
        return
    splits=get_splits(data=in_path,
                      n_splits=n_splits,
                      n_repeats=n_repeats)
    utils.make_dir(split_path)
    for i,split_i in enumerate(splits):
        split_i.save(f"{split_path}/{i}")

class AbstractClfFactory(object):
    def init(self,data):
        return self

    def __call__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def read(self,model_path:str):
        raise NotImplementedError()

    @abstractmethod
    def get_info(self)->dict:
        raise NotImplementedError()
    
#    def __repr__(self):
#        return str(self)

    @classmethod
    def get_results(cls,data_path,split_iter):
        if(type(data_path)==str):
            data=dataset.read_csv(data_path)
        else:
            data=data_path
        clf_factory=cls()
        clf_factory.init(data)
        all_results=[]
        for i,split_i in tqdm(split_iter):
            clf_i = clf_factory()
            result_i,history=split_i.eval(data,clf_i)
            all_results.append(result_i)
        return dataset.ResultGroup(all_results)

    @classmethod
    def iter_models(cls,data_path,split_iter):
        data=dataset.read_csv(data_path)
        clf_factory=cls()
        clf_factory.init(data)
        all_results=[]
        for i,split_i in tqdm(split_iter):
            clf_i = clf_factory()
            clf_i,_=split_i.fit_clf(data,clf_i)
            result_i=split_i.pred(data,clf_i)
            yield i,clf_i,result_i

class AbstractClfAdapter(object):
    @abstractmethod
    def fit(self,X,y):
        raise NotImplementedError()
    
    @abstractmethod
    def eval(self,data,split_i):
        raise NotImplementedError()

    def save(self,out_path):
        pass

    def __repr__(self):
        return str(self)