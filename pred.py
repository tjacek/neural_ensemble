import tools
tools.silence_warnings()
import argparse
import numpy as np
import tensorflow as tf 
from keras import callbacks
from collections import defaultdict
import json
import data,deep,learn

@tools.log_time(task='PRED')
def single_exp(data_path,model_path,out_path):
    clfs=['RF','SVC']
    X,y=data.get_dataset(data_path)
    pred_dict=defaultdict(lambda:[])
    for name_i,model_i,split_i in get_model_paths(model_path):
        train,test=split_i.get_dataset(X,y)
        if('base' in name_i):
            for clf_k in clfs: 
                pred_k= learn.fit_clf(train,test,clf_k,True)
                pred_dict[clf_k].append(pred_k)
        if('ens' in name_i):
            cs_train=model_i.extract(train.X)
            cs_test=model_i.extract(test.X)
            for clf_k in clfs:
                pred_k=necscf(train,test,cs_train,cs_test,clf_k)
                pred_dict[f'{name_i}({clf_k})'].append(pred_k)
        y_pred=model_i.predict(test.X)
        y_pred=np.argmax(y_pred,axis=1)
        pred_dict[name_i].append((y_pred,test.y))
    tools.make_dir(out_path)
    for name_i,pred_i in pred_dict.items():
        with open(f'{out_path}/{name_i}', 'wb') as f:
            all_pred=[ (test_i.tolist(),pred_i.tolist()) 
                        for test_i,pred_i in pred_i]
            json_str = json.dumps(all_pred, default=str)         
            json_bytes = json_str.encode('utf-8') 
            f.write(json_bytes)

def get_model_paths(model_path):
    for path_i in tools.get_dirs(model_path):
        name_i=path_i.split('/')[-1]
        for model_j in tools.get_dirs(path_i):
            if('ens' in name_i):
#                nn_j=deep.BinaryEnsemble(nn_j)
                nn_j=deep.read_ensemble(model_j)
            else:
                n_j = tf.keras.models.load_model(f'{model_j}/nn',compile=False)
            test_ind=np.load(f'{model_j}/test.npy')
            train_ind=np.load(f'{model_j}/train.npy')
            split_j=data.DataSplit(train_ind,test_ind)
            yield name_i,nn_j,split_j

def necscf(train,test,cs_train,cs_test,clf_type):
    votes=[]
    for cs_train_i,cs_test_i in zip(cs_train,cs_test):
        full_train_i=np.concatenate([train.X,cs_train_i],axis=1)
        clf_i=learn.get_clf(clf_type)
        clf_i.fit(full_train_i,train.y)
        full_test_i=np.concatenate([test.X,cs_test_i],axis=1)
        y_pred=clf_i.predict_proba(full_test_i)
        votes.append(y_pred)
    votes=np.array(votes)
    votes=np.sum(votes,axis=0)
    return np.argmax(votes,axis=1),test.y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data/cmc')
    parser.add_argument("--models", type=str, default='cmc')
    parser.add_argument("--pred", type=str, default='pred_cmc')
    parser.add_argument("--log", type=str, default='log.info')
    parser.add_argument("--dir", type=int, default=0)
    args = parser.parse_args()
    tools.start_log(args.log)
    if(args.dir>0):
        single_exp=tools.dir_fun(3)(single_exp)
    single_exp(args.data,args.models,args.pred)