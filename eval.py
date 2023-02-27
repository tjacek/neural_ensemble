import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from tensorflow import keras
import numpy as np
import json
import conf,data,learn,utils,ens_feats

def exp(data_path,model_path,clf_types=['LR','RF'],
	        ens_types=['base','common','binary']):
    raw_data=data.read_data(data_path)
    helper= get_fold_fun(raw_data,clf_types,ens_types)
    acc=[helper(path_i) 
        for path_i in data.top_files(model_path)]
    acc_dict={ id_i:[] for id_i in acc[0].keys()}
    for acc_i in acc:
        for key_j,value_j in acc_i.items():
            acc_dict[key_j].append(value_j) 
    lines=[]
    for id_i,acc_i in acc_dict.items():
    	lines.append(f'{id_i},{stats(acc_i)}')
    return lines

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

def save_lines(lines,out_path):
    f = open(out_path,"w")
    f.write('\n'.join(lines))
    f.close()

def multi_exp(data_path,model_path,out_path):
    @utils.dir_fun(as_dict=True)
    def helper(data_i):
        name_i=data_i.split('/')[-1]
        model_i=f'{model_path}/{name_i}'
        return exp(data_i,model_i,['LR','RF'],
            ['base','common','binary'])
    lines_dict=helper(data_path)
    lines=[]
    for data_i,stats_i in lines_dict.items():
        for stat_j in stats_i:
            lines.append(f'{data_i},{stat_j}')
    save_lines(lines,out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default='../uci/json')
    parser.add_argument("--models", type=str, default='../uci/models')
    parser.add_argument("--result", type=str, default='../uci/result.csv')
    args = parser.parse_args()
    multi_exp(args.json,args.models,args.result)
#    lines= exp(args.json,args.models)
#    save_lines(lines,args.result)