import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from tensorflow import keras
import numpy as np
import json
import data,learn,utils,ens_feats

def eval(data_path,model_path,clf_types=['LR','RF'],
	        ens_types=['base','common','binary']):
    raw_data=data.read_data(data_path)
    helper= get_fold_fun(raw_data,clf_types,ens_types)
    acc=[helper(path_i) 
        for path_i in data.top_files(model_path)]
#    line=stats(acc)
    print(acc)

def get_fold_fun(raw_data,clf_types,ens_types):
    ens_types=[ens_feats.get_ensemble(type_i)
        for type_i in ens_types]
    @utils.unify_cv(dir_path=None,show=False)
    def helper(in_path):
        common,binary= gen_feats(raw_data,in_path)
        acc_dir={}
        for ens_i in ens_types:
            for clf_j in clf_types:
                ens_inst=ens_i(common,binary)
                id_ij=f'{str(ens_inst)},{clf_j}'
                acc_dir[id_ij]=ens_inst(clf_j)
        return acc_dir
    return helper

def gen_feats(raw_data,in_path):
    model_path=f'{in_path}/models'
    models=[keras.models.load_model(path_i)
        for path_i in data.top_files(model_path)]  
    rename_path=f'{in_path}/rename'
    with open(rename_path, 'r') as f:
        rename_dict= json.load(f)
    common=raw_data.rename(rename_dict)
    X,y,names=common.as_dataset()
    binary=[]
    for model_i in models:
        X_i=model_i.predict(X)
        binary_i=data.from_names(X_i,names)
        binary.append(binary_i)
    return common,binary

def stats(acc,as_str=True):
    raw=[f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std]]
    if(as_str):
        return ','.join(raw)
    return raw

data_path='../../uci/json/wine'
model_path='test'
eval(data_path,model_path)