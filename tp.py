from huggingface_hub import login
#login()
from tabpfn import TabPFNClassifier
from tqdm import tqdm
import numpy as np
import base,dataset,utils

def exp(data_path,exp_path):
    data=dataset.read_csv(data_path)
    gen=base.splits_gen(exp_path,n_splits=10)

    all_results=[]
    for i,split_i in tqdm(gen):
        clf_i = TabPFNClassifier()
        result_i,history=split_i.eval(data,clf_i)
        all_results.append(result_i)
    result=dataset.ResultGroup(all_results)
    acc=np.mean(result.get_acc())
    print(f"{acc:.4f}")

#name="newthyroid"
name="wine-quality-red"
name="lymphography"
in_path=f"incr_exp/uci/data/{name}"
exp_path=f"incr_exp/uci/exp/{name}"
#print(dir(TabPFNClassifier))
exp(in_path,exp_path)