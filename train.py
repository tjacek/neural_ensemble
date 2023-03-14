import conf 
conf.silence_warnings()
import sys,os,logging,argparse
import pandas as pd 
import json,shutil,time,gzip
from tqdm import tqdm
import conf,binary,data,nn,learn,folds,utils

def multi_exp(conf):
    data.make_dir(conf['main_dict'])
    if(not conf['lazy'] and 
        (os.path.isdir(conf['model']))):
        shutil.rmtree(conf['model'])
    if(type(conf['hyper'])==str):
        get_hyper=read_hyper(conf['hyper'])
        print('Hyperparameters are loaded from {}'.format(conf['hyper']))
    else:
        get_hyper=conf['hyper']
        default= str(get_hyper(''))
        print('Default hyperparams are used {}'.format(default))
    logging.basicConfig(filename='{}_train.log'.format(conf['log']), 
        level=logging.INFO,filemode='w', 
        format='%(process)d-%(levelname)s-%(message)s')
    @utils.dir_map(depth=1)
    def helper(in_path,out_path):
        single_inter(in_path,out_path,conf,get_hyper)
    helper(conf['json'],conf['model']) 

def single_inter(in_path,out_path,conf,get_hyper):
    raw_data=data.read_data(in_path)
    data.make_dir(out_path)
    hyperparams=get_hyper(in_path)
    NeuralEnsemble=binary.get_ens(ens_type='all')
    print(f'Training models on dataset:{out_path}')
    print(f'Hyperparams:{hyperparams}')
    for i in tqdm(range(conf['n_iters'])):
        st=time.time()
        out_i=f'{out_path}/{i}'
        logging.info(f'Folder {out_i} created.')
        data.make_dir(out_i)
        folds_i=folds.make_folds(raw_data,k_folds=conf['n_split'])
        splits_i=folds.get_splits(raw_data,folds_i)
        for j,(data_j,rename_j) in enumerate(splits_i):
            ens_j= NeuralEnsemble(**hyperparams)
            learn.fit_clf(data_j,ens_j)
            out_j=f'{out_i}/{j}'
            logging.info(f'Save models {out_j}')
            save_fold(ens_j,rename_j,out_j)
        logging.info(f'Iteration {out_i} took {(time.time()-st):.4f}s')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--n_split", type=int, default=3)
    parser.add_argument("--conf",type=str,default='conf/base.cfg')
    parser.add_argument("--lazy",action='store_true')
    parser.add_argument("--default",action='store_true')
    args = parser.parse_args()
    conf_train,conf_hyper=conf.read_hyper(args.conf)
    conf_train['n_iters']=args.n_iters
    conf_train['n_split']=args.n_split
    conf_train['lazy']=args.lazy
    if(args.default):
        conf_train['hyper']=default_hyper(conf_hyper)
    multi_exp(conf_train )