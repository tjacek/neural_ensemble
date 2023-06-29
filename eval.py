import tools
tools.silence_warnings()
import argparse
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score#,f1_score
import json

def single_exp(pred_path,out_path):
    acc_dict=metric_dict(accuracy_score,pred_path)
    stats=[np.mean,np.std]
    for id_i,metric_i in acc_dict.items():
        line_i=[id_i]+[f'{stat_j(metric_i):.4}' 
                        for stat_j in stats]
        print(','.join(line_i))

def metric_dict(metric_i,pred_path):
    metric_dict={}
    for path_i in tools.top_files(pred_path):
        all_pred=read_pred(path_i)
        id_i=get_id(path_i)
        line_i=[get_id(path_i)]
        acc=[ metric_i(test_i,pred_i) 
                for test_i,pred_i in all_pred]
        metric_dict[id_i]=acc
    return metric_dict

def read_pred(path_i):
    with open(path_i, 'r') as f:        
        json_bytes = f.read()                      
        return json.loads(json_bytes)

def get_id(path_i):
    raw=path_i.split('/')
    return f'{raw[-2]},{raw[-1]}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='10_10/pred')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    if(args.dir>0):
        single_exp=tools.dir_fun(single_exp)
    single_exp(args.pred,'out')