import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn import svm
import os.path
import dataset,utils

NEURAL_CLFS=set(["MLP","TREE-MLP"])
OTHER_CLFS=set(["RF","GRAD","LR","SVM","TREE"])

class DataSplits(object):
    def __init__(self,data,splits):
        self.data=data
        self.splits=splits

    def eval(self,clf_factory):
        for split_i in self.splits:
            clf_i=clf_factory()
            split_i.fit_clf(self.data,clf_i)
            result_i=split_i.pred(self.data,clf_i)
            yield clf_i,result_i

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

    def save(self,out_path):
        return np.savez(out_path,self.train_index,self.test_index)

    def __str__(self):
        train_size=self.train_index.shape[0]
        test_size=self.test_index.shape[0]
        return f"train:{train_size},test:{test_size}"

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

class Interval(object):
    def __init__(self,start,step):
        self.start=start
        self.step=step

    def __call__(self):
        return [self.start+j 
                    for j in range(self.step)]

class DirProxy(object):
    def __init__(self,clf_type,
                      info_path,
                      clf_dict,
                      ext_dict):
        self.clf_type=clf_type
        self.info_path=info_path
        self.clf_dict=clf_dict
        self.ext_dict=ext_dict

    def make_dir(self,key):
        if(type(key)==list):
            for key_i in key:
                self.make_dir(key_i)
        else:
            utils.make_dir(self.clf_dict[key])
  
    def get_paths(self,indexes,
                       key):
        dir_path=self.clf_dict[key]
        if(indexes is None):
            return utils.top_files(dir_path),None
        if(type(indexes)==Interval):
            indexes=indexes()
        ext=self.ext_dict[key]
        paths=[ f"{dir_path}/{i}.{ext}" for i in indexes]
        return paths,indexes

    def select_paths(self,indexes,
                          key):
        paths,indexes=self.get_paths(indexes,
                                     key)
        s_paths,s_indexes=[],[]
        for i,path_i in enumerate(paths):
            if(not os.path.exists(path_i)):
                s_paths.append(path_i)
                s_indexes.append(i)
        return s_paths,s_indexes

    def path_dict(self,indexes,
                       key="models"):
        if(not key is None):
            paths,indexes=self.select_paths(indexes,
                                            key)
        return {key_i:self.get_paths(indexes,key_i)[0] 
                    for key_i in self.clf_dict}
    
    def save_info(self,clf_factory):
        utils.save_json(value=clf_factory.get_info(),
                        out_path=self.info_path)

    def read_results(self):
        return dataset.read_result_group(self.clf_dict["results"])

def get_dir_path(out_path,clf_type=None):
    if(clf_type is None):
        raw=out_path.split("/")
        clf_type=raw[-1]
        out_path="/".join(raw[:-1])
    clf_path=f"{out_path}/{clf_type}"
    utils.make_dir(clf_path)
    split_path=f"{out_path}/splits"
    info_path=f"{clf_path}/info.js"
    keys,ext_keys=["results"],["npz"]
    if(clf_type in NEURAL_CLFS):
        keys+=["models","history"]
        ext_keys+=["keras","txt"]
    clf_dict={key_i:f"{clf_path}/{key_i}" 
                for key_i in keys}
    clf_dict["splits"]=split_path
    ext_dict=dict(zip(keys,ext_keys))
    ext_dict["splits"]="npz"
    return DirProxy(clf_type=clf_type,
                    info_path=info_path,
                    clf_dict=clf_dict,
                    ext_dict=ext_dict)

def read_split(in_path):
    raw_split=np.load(in_path)
    return Split(train_index=raw_split["arr_0"],
                 test_index=raw_split["arr_1"])

def get_clf(clf_type):
    if(clf_type=="RF"): 
        return RandomForestClassifier(class_weight="balanced")
    if(clf_type=="LR"):
        return LogisticRegression(solver='liblinear')
    if(clf_type=="SVM"):
        return svm.SVC(kernel='rbf')
    if(clf_type=="GRAD"):
        return GradientBoostingClassifier()
    if(clf_type=="TREE"):
        return tree.DecisionTreeClassifier(class_weight="balanced")
    raise Exception(f"Unknow clf type:{clf_type}")

def get_splits(data_path,
               n_splits=10,
               n_repeats=1):
    data=dataset.read_csv(data_path)
    protocol=SplitProtocol(n_splits,n_repeats)
    return DataSplits(data=data,
                      splits=protocol.get_split(data))

class AbstractClfFactory(object):
    
    def init(self,data):
        pass

    def __call__(self):
        raise NotImplementedError()

    def read(self,model_path):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

class AbstractClfAdapter(object):

    def fit(self,X,y):
        raise NotImplementedError()

    def eval(self,data,split_i):
        raise NotImplementedError()

    def save(self,out_path):
        pass

class ClasicalClfFactory(AbstractClfFactory):
    def __init__(self,clf_type="RF"):
        self.clf_type=clf_type

    def __call__(self):
        return ClasicalClfAdapter(get_clf(self.clf_type))

    def read(self,model_path):
        raise Exception(f"Clasical Clf {self.clf_type} cannot be serialized ")

    def get_info(self):
        return {"clf_type":self.clf_type,"callback":None,"hyper":None}

class ClasicalClfAdapter(AbstractClfAdapter):
    def __init__(self,clf):
        self.clf=clf
    
    def fit(self,X,y):
        return self.clf.fit(X,y)

    def eval(self,data,split_i):
        return split_i.pred(data,self.clf)