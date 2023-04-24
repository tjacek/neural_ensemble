import tools
tools.silence_warnings()
import numpy as np
import argparse
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from time import time
import clfs,ens,learn

def single_exp(data_path,n_splits,n_repeats,
    hyper_path,ens_type,out_path='out',verbose=False):
#    hyper_dict=parse_hyper(hyper_path)
    df=pd.read_csv(data_path) 
    X,y=tools.prepare_data(df)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, 
        n_repeats=n_repeats, random_state=4)
    clf=clfs.get_ens(ens_type,hyper=None)
    tools.make_dir(out_path)
    for i,split_i in enumerate(cv.split(X,y)):
        clf_i= train_model(X,y,clf,split_i,verbose)
        out_i=f'{out_path}/{i}'
        clfs.save_clf(clf_i,out_i)
        train_i,test_i=split_i
        np.save(f'{out_i}/train', train_i)
        np.save(f'{out_i}/test', test_i)

def train_model(X,y,clf_i,split_i,verbose=True):
    start=time()
    train_index,test_index=split_i
    X_i,y_i= X[train_index],y[train_index]
    if(ens.is_neural_ensemble(clf_i)):
        clf_i= clf_i.fit(X_i,y_i,verbose)
    else:
        clf_i= clf_i.fit(X_i,y_i)
    end=time()
    print(f'Training time-{str(clf_i)}:{(end-start):.2f}s')
    return clf_i
#    print(clfs.get_desc(clf_i))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/wine-quality-red')
    parser.add_argument("--hyper", type=str, default='-')
    parser.add_argument("--ens", type=str, default='CPUClf_2')
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--verbose",action='store_true')
    args = parser.parse_args()
    if(args.hyper=='-'):
        args.hyper=None
    return args

if __name__ == '__main__':
    args=parse_args()
    single_exp(args.data,args.n_splits,args.n_repeats,
        hyper_path=args.hyper,ens_type=args.ens,verbose=args.verbose)