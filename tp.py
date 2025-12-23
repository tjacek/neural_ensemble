from huggingface_hub import login
#login()
from tabpfn import TabPFNClassifier
from tqdm import tqdm
import numpy as np
import base,dataset,utils

class TabpfFactory(base.AbstractClfFactory):
    def init(self,data):
        return self

    def __call__(self):
        return TabPFNClassifier()

    def read(self,model_path:str):
        pass

    def get_info(self)->dict:
        return {"clf_type":"TabPFN","callback":"-",
                "hyper":"-"}
    

def exp(data_path,exp_path,n_splits=30):
    data=dataset.read_csv(data_path)
    gen=base.splits_gen(exp_path,n_splits=n_splits)
    clf_factory=TabpfFactory()
    clf_factory.init(data)
    tab_path=f"{exp_path}/TabPF"
    utils.make_dir(tab_path)
    all_results=[]
    for i,split_i in tqdm(gen):
        clf_i = clf_factory()
        result_i,history=split_i.eval(data,clf_i)
        all_results.append(result_i)
    result=dataset.ResultGroup(all_results)
    result.save(f"{tab_path}/results")
    acc=np.mean(result.get_acc())
    print(f"{acc:.4f}")

def basic_exp():
    names=["cmc"]
    data="uci"
    for name_i in names:
        in_path=f"incr_exp/{data}/data/{name_i}"
        exp_path=f"incr_exp/{data}/exp/{name_i}"
        exp(in_path,exp_path)

basic_exp()