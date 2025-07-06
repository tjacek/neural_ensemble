import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import dataset

NEURAL_CLFS=set(["MLP"])

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

class ClfFactory(object):
    def __init__(self,hyper_params=None,
                      loss_gen=None):
        if(hyper_params is None):
           hyper_params=default_hyperparams()
        self.params=None
        self.hyper_params=hyper_params
        self.class_dict=None
        self.loss_gen=loss_gen
    
    def init(self,data):
        self.params={'dims': (data.dim(),),
                     'n_cats':data.n_cats(),
                     'n_epochs':1000}
        self.class_dict=dataset.get_class_weights(data.y)

    def __call__(self):
        raise NotImplementedError()

    def read(self,model_path):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

class ClfAdapter(object):
    def __init__(self, params,
                       hyper_params,
                       class_dict=None,
                       model=None,
                       loss_gen=None,
                       verbose=0):
        self.params=params
        self.hyper_params=hyper_params
        self.class_dict=class_dict
        self.model = model
        self.loss_gen=loss_gen
        self.verbose=verbose

    def fit(self,X,y):
        raise NotImplementedError()

    def eval(self,data,split_i):
        test_data_i=data.selection(split_i.test_index)
        raw_partial_i=self.partial_predict(test_data_i.X)
        result_i=dataset.PartialResults(y_true=test_data_i.y,
                                        y_partial=raw_partial_i)
        return result_i

    def save(self,out_path):
        raise NotImplementedError()

def default_hyperparams():
    return {'layers':2, 'units_0':2,
            'units_1':1,'batch':False}
