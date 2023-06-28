import tools
tools.silence_warnings()
import argparse
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score#,f1_score
import json

def single_exp(pred_path,out_path):
    metrics=[accuracy_score,balanced_accuracy_score]
    stats=[np.mean,np.std]
    for path_i in tools.top_files(pred_path):
        all_pred=read_pred(path_i)
        line_i=[f'{path_i}']
        for metric_j in metrics:
            acc=[ metric_j(test_i,pred_i) 
                    for test_i,pred_i in all_pred]
            for stat_k in stats:
                line_i.append(f'{stat_k(acc):.4}')
        print(','.join(line_i))

def read_pred(path_i):
    with open(path_i, 'r') as f:        
        json_bytes = f.read()                      
        return json.loads(json_bytes)
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='10_10/pred')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    if(args.dir>0):
        single_exp=tools.dir_fun(single_exp)
    single_exp(args.pred,'out')