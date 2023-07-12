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
    clfs=['RF','SVC']
    X,y=data.get_dataset(data_path)
    pred_dict=defaultdict(lambda:[])
    for name_i,exp_i in get_exps(model_path):
        train,test=exp_i.get_features(X,y)
        for clf_j in clfs:
            if(exp_i.is_ens()):
                y_pred=necscf(train,test,clf_j)
                pred_dict[f'{name_i}-{clf_j}'].append((y_pred,test.y))
            else:
            	y_pred=single_clf(train,test,clf_j)
            	pred_dict[clf_j].append((y_pred,test.y))
        y_pred=exp_i.model.predict(test.X)
        y_pred=np.argmax(y_pred,axis=1)
        pred_dict[name_i].append((y_pred,test.y))
    tools.make_dir(out_path)
    for name_i,pred_i in pred_dict.items():
        save_pred(f'{out_path}/{name_i}',pred_i)

def save_pred(out_path,pred_i):        
    with open(out_path, 'wb') as f:
        all_pred=[ (test_i.tolist(),pred_i.tolist()) 
                        for test_i,pred_i in pred_i]
        json_str = json.dumps(all_pred, default=str)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

def get_exps(model_path):
    for path_i in tools.get_dirs(model_path):
        name_i=path_i.split('/')[-1]
        for exp_j in tools.get_dirs(path_i):
        	yield name_i,train.read_exp(exp_j)

def necscf(train,test,clf_type):
    votes=[]
    for cs_train_i,cs_test_i in zip(train.cs,test.cs):
        full_train_i=np.concatenate([train.X,cs_train_i],axis=1)       
        clf_i=learn.get_clf(clf_type)
        clf_i.fit(full_train_i,train.y)
        full_test_i=np.concatenate([test.X,cs_test_i],axis=1)
        y_pred=clf_i.predict_proba(full_test_i)
        votes.append(y_pred)
#    if(votes):
#        return votes
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1)

def single_clf(train,test,clf_type):
    clf_i=learn.get_clf(clf_type)
    clf_i.fit(train.X,train.y)
    return clf_i.predict(test.X)

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