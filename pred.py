import tools
tools.silence_warnings()
import argparse
import numpy as np
import tensorflow as tf 
from keras import callbacks
from collections import defaultdict
import json
import data,deep,learn,train

@tools.log_time(task='PRED')
def single_exp(data_path,model_path,out_path):
    clfs=['RF']#,'SVC']
    X,y=data.get_dataset(data_path)
    pred_dict=defaultdict(lambda:[])
    for name_i,exp_path_i in exp_paths(model_path):
        exp_i=train.read_exp(exp_path_i)
        print(exp_i.is_ens())
        print(exp_path_i)

def exp_paths(model_path):
    for path_i in tools.get_dirs(model_path):
        name_i=path_i.split('/')[-1]
        for exp_j in tools.get_dirs(path_i):
        	yield name_i,exp_j

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../data')
    parser.add_argument("--models", type=str, default='../test3/models')
    parser.add_argument("--pred", type=str, default='../test3/pred')
    parser.add_argument("--log", type=str, default='../test3/log.info')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    tools.start_log(args.log)
    if(args.dir>0):
        single_exp=tools.dir_fun(3)(single_exp)
    single_exp(args.data,args.models,args.pred)