import tools
tools.silence_warnings()
import argparse
import numpy as np
import tensorflow as tf 
from keras import callbacks
from collections import defaultdict
import json,os
from tqdm import tqdm
import data,deep,learn,exp,variants

@tools.log_time(task='PRED')
def single_exp(data_path,model_path,out_path):
    print(data_path)
    clfs=['RF','SVC']
    single_var=[variants.NoEnsemble(clfs)]
    ens_var   =[variants.BasicVariant(clfs),
                variants.BinaryVariant(clfs+['LR'])]
    X,y=data.get_dataset(data_path)
    pred_dict=defaultdict(lambda:[])
    for name_i,exp_i in tqdm(get_exps(model_path)):
        train,test=exp_i.get_features(X,y)
        if(exp_i.is_ens()):
            for var_j in ens_var:
                for id_k,pred_k in var_j(train,test):
                    pred_dict[f'{name_i}-{id_k}'].append((pred_k,test.y))
        else:
            for var_j in single_var:
                for id_k,pred_k in var_j(train,test):
                    pred_dict[f'{name_i}-{id_k}'].append((pred_k,test.y))
    tools.make_dir(out_path)
    for name_i,pred_i in pred_dict.items():
        save_pred(f'{out_path}/{name_i}',pred_i)

def save_pred(out_path,pred_i):        
    with open(out_path, 'wb') as f:
        all_pred=[]
        for test_i,pred_i in pred_i:
            if(type(test_i)!=list):
                test_i=test_i.tolist()
            if(type(pred_i)!=list):
                pred_i=pred_i.tolist()
            all_pred.append((test_i,pred_i))                
        json_str = json.dumps(all_pred, default=str)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

def get_exps(model_path):
    for path_i in tools.get_dirs(model_path):
        name_i=path_i.split('/')[-1]
        for exp_j in tools.get_dirs(path_i):
        	yield name_i,exp.read_exp(exp_j)

if __name__ == '__main__':
    dir_path='../optim_alpha/s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../s_uci')
    parser.add_argument("--models", type=str, default=f'{dir_path}/models')
    parser.add_argument("--pred", type=str, default=f'{dir_path}/pred')
    parser.add_argument("--log", type=str, default=f'{dir_path}/log.info')
#    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    tools.start_log(args.log)
    if(os.path.isdir(args.data)):
        single_exp=tools.dir_fun(3)(single_exp)
    single_exp(args.data,args.models,args.pred)