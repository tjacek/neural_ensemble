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
#from sklearn.model_selection import RepeatedStratifiedKFold
#from sklearn.model_selection import GridSearchCV
#from skopt import BayesSearchCV
import json,shutil
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


def train_exp(conf_dict):
    if(not conf_dict['lazy'] and 
        (os.path.isdir(conf_dict['model']))):
        shutil.rmtree(conf_dict['model'])
    set_logging(conf_dict['log'])
    if(conf_dict['single']):
        fun=gen_data
    else:
        fun=multi_exp
    hyper_optim=parse_hyper(conf_dict)
    fun(conf_dict['json'],conf_dict['model'],
        n_iters=conf_dict['n_iters'],n_split=conf_dict['n_split'],
        hyper_optim=hyper_optim,ens_type="all")

def set_logging(log_path):
    logging.basicConfig(filename=log_path, 
        level=logging.INFO,filemode='w', 
        format='%(process)d-%(levelname)s-%(message)s')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--n_split", type=int, default=3)
    parser.add_argument("--conf",type=str,default='conf/grid.cfg')
    parser.add_argument("--lazy",action='store_true')
    parser.add_argument("--single",action='store_true') 
    args = parser.parse_args()
    conf_dict=conf.read_conf(args.conf)
    conf_dict['n_iters']=args.n_iters
    conf_dict['n_split']=args.n_split
    conf_dict['lazy']=args.lazy
    conf_dict['single']=args.single
#    raise Exception(conf_dict)
    train_exp(conf_dict)