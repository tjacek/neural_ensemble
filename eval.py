import tools
import argparse
import numpy as np
from sklearn.metrics import accuracy_score#,f1_score
import json

def single_exp(pred_path):
    for path_i in tools.top_files(pred_path):
        all_pred=read_pred(path_i)
        acc=[ accuracy_score(test_i,pred_i) 
                 for test_i,pred_i in all_pred]
        print(f'{path_i},{np.mean(acc):.4},{np.std(acc):.4}')

def read_pred(path_i):
    with open(path_i, 'r') as f:        
        json_bytes = f.read()                      
        return json.loads(json_bytes)
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='pred')
    args = parser.parse_args()
    single_exp(args.pred)