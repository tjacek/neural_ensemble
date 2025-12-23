from huggingface_hub import login
#login()
from tabpfn import TabPFNClassifier
from tqdm import tqdm
import pickle
import numpy as np
import base,clfs,dataset,utils

class TabpfFactory(base.AbstractClfFactory):
    def init(self,data):
        return self

    def __call__(self):
        raw_clf=TabPFNClassifier()
        return TabpfAdapter(raw_clf)

    def read(self,model_path:str):
        pass

    def get_info(self)->dict:
        return {"clf_type":"TabPFN","callback":"-",
                "hyper":"-"}
    @classmethod
    def get_id(self):
        return "TabPF"

class TabpfAdapter(base.AbstractClfAdapter):
    def __init__(self,tab_model):
        self.tab_model=tab_model
    
    def predict(self,X):
        return self.tab_model.predict(X)
    
    def fit(self,X,y):
        self.tab_model.fit(X,y)    

    def eval(self,data,split_i):
        raise NotImplementedError()

    def save(self,out_path):
        with open(out_path, 'wb') as f:
            pickle.dump(self.tab_model, f)

    def __str__(self):
        return "TabPF"

class TreeTabpfFactory(base.AbstractClfFactory):
    def __init__(self,feature_params=None):
        if(feature_params is None):
            feature_params={"tree_factory":"random",
                            "extr_factory":("info",30),
                            "concat":True}
        self.feature_params=feature_params

    def init(self,data):
        return self

    def __call__(self):
        extractor_factory=clfs.FeatureExtactorFactory(**self.feature_params)
        return TreeTabpfAdapter(extractor_factory)

    def read(self,model_path:str):
        pass

    def get_info(self):
        return {"clf_type":"TreeTabPF","callback":"-",
                "hyper":"-",
                "feature_params":self.feature_params}

    @classmethod
    def get_id(self):
        return "TreeTabPF"

class TreeTabpfAdapter(base.AbstractClfAdapter):
    def __init__(self,extractor_factory):
        self.extractor_factory=extractor_factory
        self.tab_model=None
        self.extractor=None
    
    def fit(self,X,y):
        self.tab_model=TabPFNClassifier()
        self.extractor=self.extractor_factory(X,y)
        new_X=self.extractor(X)
        self.tab_model.fit(new_X,y)  
    
    def predict(self,X):
        new_X=self.extractor(X)
        return self.tab_model.predict(new_X)

    def eval(self,data,split_i):
        raise NotImplementedError()

    def save(self,out_path):
        utils.make_dir(out_path)
        self.extractor.extractor.save(out_path)

    def __str__(self):
        return "TreeTabPF"

def exp(data_path,exp_path,n_splits=30):
    factory_type=TreeTabpfFactory
    gen=base.splits_gen(exp_path,n_splits=n_splits)
    tab_path=f"{exp_path}/{factory_type.get_id()}"
    utils.make_dir(tab_path)
    all_results=[]
    result=factory_type.get_results(data_path,gen)
    result.save(f"{tab_path}/results")
    acc=np.mean(result.get_acc())
    print(f"{acc:.4f}")

def exp_save(data_path,exp_path,n_splits=30):
    factory_type=TreeTabpfFactory
    gen=base.splits_gen(exp_path,n_splits=n_splits)
    tab_path=f"{exp_path}/{factory_type.get_id()}"
    utils.make_dir(tab_path)
    model_path=f"{tab_path}/models"
    utils.make_dir(model_path)
    all_results=[]
    model_iter=factory_type.iter_models(data_path,gen)
    for i,model_i,result_i in model_iter:
        model_i.save(f"{model_path}/{i}")
        all_results.append(result_i)
    result=dataset.ResultGroup(all_results)    
    result.save(f"{tab_path}/results")
    acc=np.mean(result.get_acc())
    print(f"{acc:.4f}")


def multi_exp():
    names=["vehicle"]
    data="uci"
    for name_i in names:
        in_path=f"incr_exp/{data}/data/{name_i}"
        exp_path=f"incr_exp/{data}/exp/{name_i}"
        exp_save(in_path,exp_path)

multi_exp()