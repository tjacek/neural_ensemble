import conf 
conf.silence_warnings()
import sys,os,logging,argparse
import pandas as pd 
import json,shutil,time,gzip
from tqdm import tqdm
import conf,binary,data,nn,learn,folds,utils

def multi_exp(conf_dict):
    data.make_dir(conf_dict['main_dir'])
    if(not conf_dict['lazy'] and 
        (os.path.isdir(conf_dict['model']))):
        shutil.rmtree(conf_dict['model'])
    if(type(conf_dict['hyper'])==str):
        get_hyper=read_hyper(conf_dict['hyper'])
        print('Hyperparameters are loaded from {}'.format(conf_dict['hyper']))
    else:
        get_hyper=conf_dict['hyper']
        default= str(get_hyper(''))
        print('Default hyperparams are used {}'.format(default))
    logging.basicConfig(filename='{}_train.log'.format(conf_dict['log']), 
        level=logging.INFO,filemode='w', 
        format='%(process)d-%(levelname)s-%(message)s')
    @utils.dir_map(depth=1)
    def helper(in_path,out_path):
        single_iter(in_path,out_path,conf_dict,get_hyper)
    helper(conf_dict['data_dir'],conf_dict['model']) 

def single_iter(in_path,out_path,conf_dict,get_hyper):
    raw_data=data.read_data(in_path)
    data.make_dir(out_path)
    hyperparams=get_hyper(in_path)
    NeuralEnsemble=binary.get_ens(conf_dict['binary_type'])
    print(f'Training models on dataset:{out_path}')
    print(f'Hyperparams:{hyperparams}')
    for i in tqdm(range(conf_dict['n_iters'])):
        st=time.time()
        out_i=f'{out_path}/{i}'
        logging.info(f'Folder {out_i} created.')
        data.make_dir(out_i)
        folds_i=folds.make_folds(raw_data,k_folds=conf_dict['n_splits'])
        splits_i=folds.get_splits(raw_data,folds_i)
        for j,(data_j,rename_j) in enumerate(splits_i):
            ens_j= NeuralEnsemble(**hyperparams)
            learn.fit_clf(data_j,ens_j)
            out_j=f'{out_i}/{j}'
            conf.log_time(f'Save models {out_j}',st)
            save_fold(ens_j,rename_j,out_j)
        conf.log_time(f'Iteration {out_i}',st)
    print(f'Models saved at {out_path}\n')

def save_fold(ens_j,rename_j,out_j):
    raw_dict={
      'rename':rename_j,
      'models':[extr.to_json() 
            for extr in ens_j.extractors]
    }
    with gzip.open(out_j, 'wb') as f:
        json_str = json.dumps(raw_dict)         
        json_bytes = json_str.encode('utf-8') 
        f.write(json_bytes)

def read_hyper(in_path):
    hyper_frame=pd.read_csv(in_path)
    hyper_dict={}
    for i,row_i in hyper_frame.iterrows():
        dict_i= row_i.to_dict()
        data_i=dict_i['dataset']
        del dict_i['dataset']
        hyper_dict[data_i]=dict_i
    def helper(path_i):
        return hyper_dict[path_i.split('/')[-1]]
    return helper

def default_hyper(conf_hyper):
    names=conf_hyper['hyperparams']
    def helper(name_i):
        value_i=conf_hyper[f'default_{name_i}']
        if(value_i.isnumeric()):
            return int(value_i)
        return value_i
    hyper_dict={name_i:helper(name_i) 
                    for name_i in names}
    return lambda path: hyper_dict

def parse_args(default_conf='conf/l1.cfg'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--conf",type=str,default=default_conf)
    parser.add_argument("--data_dir",type=str)
    parser.add_argument("--main_dir",type=str)
    parser.add_argument("--batch_size",type=int)
    parser.add_argument("--default",action='store_true')
    parser.add_argument("--lazy",action='store_true')
    parser.add_argument("--datasets",type=str,default='-')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=parse_args(default_conf='conf/l1.cfg')
    conf_dict=conf.read_conf(args.conf,
        ['hyper','clf'])#,args.dir_path)
    conf.add_dir_paths(conf_dict,args.data_dir,
                                 args.main_dir)
    if(args.default):
        conf_train['hyper']=default_hyper(conf_train)     
    conf_dict['n_iters']=args.n_iters
    conf_dict['n_splits']=args.n_splits
    conf_dict['lazy']=args.lazy
    conf.GLOBAL['batch_size']=args.batch_size
    if(args.datasets!='-'):
        datasets=args.datasets.strip().split(',')
        datasets=[ '{}/{}'.format(conf_dict['data_dir'],data_i) 
                for data_i in datasets]
        conf_dict['data_dir']=datasets
    print(conf_dict)
    multi_exp(conf_dict)