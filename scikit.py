import tools
tools.silence_warnings()
import argparse
import numpy as np
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import data,deep,learn,train

class ScikitAdapter(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=0.5, 
                       hyper=None,
                       ens_type='weighted',
                       clf_type='RF'):
        self.alpha = alpha
        self.hyper = hyper
        self.ens_type=ens_type
        self.clf_type=clf_type
        self.neural_ensemble=None 
        self.clfs=[]

    def fit(self,X,targets):
        ens_factory=deep.get_ensemble((self.ens_type,self.alpha))
        params=data.get_dataset_params(X,targets) 
        self.neural_ensemble=ens_factory(params,
                                         self.hyper)
        self.neural_ensemble.fit(X,targets)
        full=self.neural_ensemble.get_full(X) #.extract(X)
        if(len(self.clfs)>0):
            self.clfs=[]
        for full_i in full:
            clf_i=learn.get_clf(self.clf_type)
            clf_i.fit(full_i,targets)
            self.clfs.append(clf_i)

    def predict_proba(self,X):
        full=self.neural_ensemble.get_full(X)
        votes=[clf_i.predict_proba(full_i) 
             for clf_i,full_i in zip(self.clfs,full)]
        votes=np.array(votes)
        return np.sum(votes,axis=0)

    def predict(self,X):
        prob=self.predict_proba(X)
        return np.argmax(prob,axis=1)

def alpha_exp(data_path,hyper_path,n_split,n_repeats,out_path):
    hyper_df=pd.read_csv(hyper_path)
    @tools.log_time(task='ALPHA')
    def helper(data_i):
        X,y=data.get_dataset(data_i)
        name_i=data_i.split('/')[-1]
        hyper_i=hyper_df[hyper_df['dataset']==name_i]
        hyper_i=hyper_i.iloc[0].to_dict()
        print(hyper_i)
    if(os.path.isdir(data_path)):
        helper=tools.dir_fun(2)(helper)
    helper(data_path)    

if __name__ == "__main__":
    dir_path='../optim_alpha/s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../data')
    parser.add_argument("--hyper", type=str, default=f'{dir_path}/hyper.csv')
    parser.add_argument("--n_split", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--out_path", type=str, default='alpha.csv')
    parser.add_argument("--log", type=str, default='log')
#    parser.add_argument("--dir", type=int, default=1)
    args = parser.parse_args()
    tools.start_log(args.log)
#    if(os.path.isdir( args.dir)):
#        multi_exp(args,args.out_path)
#    else:
    alpha_exp(args.data,
              args.hyper,
              args.n_split,
              args.n_iter,
              args.out_path)