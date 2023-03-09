import sys
import os
import sys
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import json
from tqdm import tqdm
import conf,binary,data,nn,learn,folds,utils

def multi_exp(in_path,out_path,n_iters=10,n_split=10,hyper_optim=False,
        ens_type="all"):
    @utils.dir_map(depth=1)
    def helper(in_path,out_path):
        gen_data(in_path,out_path,n_iters,n_split,hyper_optim,ens_type)
    helper(in_path,out_path) 

def gen_data(in_path,out_path,n_iters=10,n_split=10,hyper_optim=None,
                ens_type="all"):
    raw_data=data.read_data(in_path)
    data.make_dir(out_path)
    hyperparams=hyper_optim(raw_data,ens_type,n_split)
    NeuralEnsemble=binary.get_ens(ens_type)
    print(f'Training models {out_path}')
    for i in tqdm(range(n_iters)):
        out_i=f'{out_path}/{i}'
        data.make_dir(out_i)
        folds_i=folds.make_folds(raw_data,k_folds=n_split)
        splits_i=folds.get_splits(raw_data,folds_i)
        for j,(data_j,rename_j) in enumerate(splits_i):
            ens_j= NeuralEnsemble(**hyperparams)
            learn.fit_clf(data_j,ens_j)
            out_j=f'{out_i}/{j}'
            save_fold(ens_j,rename_j,out_j)

def save_fold(ens_j,rename_j,out_j):
    logging.info(f'Save models {out_j}')
    data.make_dir(out_j)
    ens_j.save(f'{out_j}/models')
    with open(f'{out_j}/rename', 'wb') as f:
        json_str = json.dumps(rename_j)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

class HyperOptimisation(object):
    def __init__(self,default_params=None,search_spaces=None,verbosity=True):
        if(default_params is None):
            default_params={'n_hidden':250,'n_epochs':200}
        if(search_spaces is None):
            search_spaces={'n_hidden':[25,50,100,200],
                    'n_epochs':[100,200,300,500]}
        self.default_params=default_params
        self.search_spaces=search_spaces
        self.search_alg= GridOptim() #BayesOptim()
#        self.verbosity=verbosity

    def __call__(self,train,ensemble=None,n_split=10):
        if(self.search_spaces):
            print('Optimisation of hyperparams')
            return self.optim(train,ensemble,n_split)
        else:
            return {'n_hidden':250,'n_epochs':200}

    def optim(self,train,ensemble=None,n_split=10):
        if(type(train)==str):
            train=data.read_data(train)
        ensemble=binary.get_ens(ensemble)
        def helper(X_train,y_train):
            cv_gen=RepeatedStratifiedKFold(n_splits=n_split, 
                    n_repeats=1, random_state=1)
            search=self.search_alg((X_train,y_train),ensemble(),
                self.search_spaces,cv_gen)
            best_estm=search.best_estimator_
            return best_estm.get_params(deep=True)
        train_tuple=train.as_dataset()[:2]
        best_params= helper(*train_tuple)
        return best_params

class BayesOptim(object):
    def __init__(self,n_jobs=5,verbosity=True):
        self.n_jobs=n_jobs
        self.verbosity=verbosity

    def __call__(self,train_tuple,clf,search_spaces,cv_gen):
        search = BayesSearchCV(estimator=clf,verbose=0,
                    search_spaces=search_spaces,n_jobs=self.n_jobs,cv=cv_gen)
        X_train,y_train=train_tuple
        search.fit(X_train,y_train,callback=self.get_callback()) 
        return search

    def get_callback(self):
        if(self.verbosity):
            count=0
            def callback(optimal_result):
                nonlocal count
                print(f"Iteration {count}")
                print(f"Best params so far {optimal_result.x}")
                count+=1
            return callback
        return None

class GridOptim(object):
    def __init__(self,n_jobs=5):
        self.n_jobs=n_jobs

    def __call__(self,train_tuple,clf,search_spaces,cv_gen):
        search = GridSearchCV(estimator=clf,param_grid=search_spaces,
#                  verbose=0,
                  n_jobs=self.n_jobs,cv=cv_gen)
        X_train,y_train=train_tuple
        search.fit(X_train, y_train)
        return search

def parse_bayes(args):
    train_conf=conf.read_conf(args.conf)
    verbosity=train_conf.getboolean('verbosity')
    bayes=  False if(args.no_bayes) else None
    return HyperOptimisation(search_spaces=bayes,verbosity=verbosity)

def set_logging(train_conf):
    logging.basicConfig(filename=train_conf['log'], 
        level=logging.INFO,filemode='w', 
        format='%(process)d-%(levelname)s-%(message)s')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--n_split", type=int, default=3)
    parser.add_argument("--conf",type=str,default='conf/base.cfg')
    parser.add_argument("--no_bayes",action='store_true')
    parser.add_argument("--single",action='store_true')
    args = parser.parse_args()
    train_conf=conf.read_conf(args.conf)
    set_logging(train_conf)
    bayes=parse_bayes(args)
    if(args.single):
        fun=gen_data
    else:
        fun=multi_exp
    fun(train_conf['json'],train_conf['model'],
        n_iters=args.n_iters,n_split=args.n_split,
        hyper_optim=bayes,ens_type="all")