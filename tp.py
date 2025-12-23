from huggingface_hub import login
#login()
from tabpfn import TabPFNClassifier
from tqdm import tqdm
import pickle
import numpy as np
import base,dataset,utils

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

class TabpfAdapter(object):
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

def exp(data_path,exp_path,n_splits=30):
    gen=base.splits_gen(exp_path,n_splits=n_splits)
    tab_path=f"{exp_path}/TabPF"
    utils.make_dir(tab_path)
    all_results=[]
    result=TabpfFactory.get_results(data_path,gen)
    result.save(f"{tab_path}/results")
    acc=np.mean(result.get_acc())
    print(f"{acc:.4f}")

def exp_save(data_path,exp_path,n_splits=30):
    gen=base.splits_gen(exp_path,n_splits=n_splits)
    tab_path=f"{exp_path}/TabPF"
    utils.make_dir(tab_path)
    model_path=f"{tab_path}/models"
    utils.make_dir(model_path)
    all_results=[]
    model_iter=TabpfFactory.iter_models(data_path,gen)
    for i,model_i,result_i in model_iter:
        model_i.save(f"{model_path}/{i}")
        all_results.append(result_i)
    result=dataset.ResultGroup(all_results)    
    result.save(f"{tab_path}/results")
    acc=np.mean(result.get_acc())
    print(f"{acc:.4f}")


def multi_exp():
    names=["dna"]
    data="multi"
    for name_i in names:
        in_path=f"incr_exp/{data}/data/{name_i}"
        exp_path=f"incr_exp/{data}/exp/{name_i}"
        exp_save(in_path,exp_path)

multi_exp()