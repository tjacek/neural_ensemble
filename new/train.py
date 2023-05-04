import tools
tools.silence_warnings()
import numpy as np
import argparse
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from time import time
import clfs,ens,learn,models

def single_exp(data_path,n_splits,n_repeats,ens_type,
        hyper_path,out_path='out',verbose=False):
    if(ens_type in set(['best','CPU','GPU'])):
        hyper_dict=parse_hyper(hyper_path,ens_type)
        ens_type=list(hyper_dict.keys())[0]
    else:
        hyper_dict=parse_hyper(hyper_path,None)
    df=pd.read_csv(data_path) 
    X,y=tools.prepare_data(df)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, 
        n_repeats=n_repeats, random_state=4)
    models_io=models.ModelIO(out_path)
    for i,split_i,train_i,test_i in models.split_iterator(cv,X,y):
        clf=clfs.get_ens(ens_type,hyper=hyper_dict[ens_type])
        clf_i= train_model(train_i[0],train_i[1],clf,verbose)
        models_io.save(clf_i,i,split_i)
    
def train_model(X_i,y_i,clf_i,verbose=True):
    start=time()
    if(ens.is_neural_ensemble(clf_i)):
        clf_i= clf_i.fit(X_i,y_i,verbose)
    else:
        clf_i= clf_i.fit(X_i,y_i)
    end=time()
    print(f'Training time-{str(clf_i)}:{(end-start):.2f}s')
    return clf_i

def parse_hyper(hyper_path,selection=None):
    if(hyper_path is None):
        hyper_dict={clf_i:None for clf_i in clfs.CLFS_NAMES}
        return hyper_dict
    hyper_dict,score_dict={},{}
    with open(hyper_path) as f:
        lines = f.readlines()[1:]
        for line_i in lines:
            if('data' in line_i):
                continue
            raw= line_i.split(',')
            clf,hyper,score=raw[0],raw[1:-1],float(raw[-1])
            if(selection=='GPU' or selection=='CPU'):
                if(selection in clf ):
                    continue
            hyper_dict[clf]=eval(','.join(hyper))
            score_dict[clf]=score
    if(not (selection is None)):
       clf=list(hyper_dict.keys())
       score=[score_dict[clf_i] for clf_i in clf]
       best=clf[np.argmax(score)]
       return {best:hyper_dict[best]}
    return hyper_dict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='uci/wine-quality-red')
    parser.add_argument("--hyper", type=str, default='-')
    parser.add_argument("--ens", type=str, default='CPUClf_1')
    parser.add_argument("--out", type=str, default='out')
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--verbose",action='store_true')
    parser.add_argument("--log_path", type=str, default='log.time')

    args = parser.parse_args()
    if(args.hyper=='-'):
        args.hyper=None
    return args

if __name__ == '__main__':
    args=parse_args()
    if(args.hyper=='-'):
        args.hyper=None
    tools.start_log(args.log_path)
    start=time()
    single_exp(args.data,args.n_splits,args.n_repeats,ens_type=args.ens,
        hyper_path=args.hyper,out_path=args.out,verbose=args.verbose) 
    tools.log_time(f'TRAIN:{args.data}',start) 