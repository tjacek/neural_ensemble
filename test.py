import conf 
conf.silence_warnings()
import sys,os,logging,argparse
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
import pandas as pd 
import json,time,gzip,shutil
import conf,data,learn,utils,ens_feats
from tqdm import tqdm

def test_exp(conf):
#    if(os.path.isdir(conf['output'])):
#        shutil.rmtree(conf['output'])
    logging.basicConfig(filename='{}_test.log'.format(conf['log']), 
        level=logging.INFO,filemode='w', 
        format='%(process)d-%(levelname)s-%(message)s')
    @utils.dir_map(1,overwrite=True)
    def helper(model_path,output_path):
        name=model_path.split('/')[-1]
        data_path='{}/{}'.format(conf['data_dir'],name)
        return exp(data_path,model_path,output_path,conf)
    helper(conf['model'],conf['output'])#)

def exp(data_path,model_path,output_path,conf):
    raw_data=data.read_data(data_path)
    clf_types,ens_types=conf['clf_types'],conf['ens_types']
    print(f'\nTest models on dataset:{data_path}')
    print('Multiclass classifiers types used:{}'.format(','.join(clf_types)))
    print('Ensemble variant used:{}'.format(','.join(ens_types)))
    helper= get_fold_fun(raw_data,clf_types,ens_types)
    results=[helper(path_i) 
        for path_i in tqdm(data.top_files(model_path))]
    data.make_dir(output_path)
    for id_i in results[0].keys():
        data.make_dir(f'{output_path}/{id_i}')
    for i,result_i in enumerate(results):
        for key_j,value_j in result_i.items():
            value_j.save(f'{output_path}/{key_j}/{i}') 
    print('Saved result for {} at {}\n'.format(data_path,output_path))

def get_fold_fun(raw_data,clf_types,ens_types):
    ens_types=[ens_feats.get_ensemble(type_i)
        for type_i in ens_types]
    @utils.unify_cv(dir_path=None,show=False)
    def helper(in_path):
        st=time.time()
        logging.info(f'Read models:{in_path}')
        common,binary= gen_feats(raw_data,in_path)
        conf.log_time(f'Features generated:{in_path}',st)
        result_dir={}
        for ens_i in ens_types:
            for clf_j in clf_types:
                ens_inst=ens_i(common,binary)
                id_ij=f'{str(ens_inst)}_{clf_j}'
                result_dir[id_ij]=ens_inst(clf_j)
        conf.log_time(f'Evaluate models from:{in_path}',st)
        return result_dir
    return helper

def gen_feats(raw_data,in_path):
    rename_dict,models=read_fold(in_path)
    common=raw_data.rename(rename_dict)
    X,y,names=common.as_dataset()
    binary=[]
    for model_i in models:
        X_i=model_i.predict(X)
        binary_i=data.from_names(X_i,names)
        binary.append(binary_i)
    return common,binary

def read_fold(in_path):
    with gzip.open(in_path, 'r') as f:        
        json_bytes = f.read()                      
        json_str = json_bytes.decode('utf-8')           
        raw_dict = json.loads(json_str)
        rename_dict=raw_dict['rename']
        models=[ model_from_json(model_i) 
            for model_i in raw_dict['models']]
        return rename_dict,models

def parse_args(default_conf='conf/l1.cfg'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",type=str,default=default_conf)
    parser.add_argument("--data_dir",type=str)
    parser.add_argument("--main_dir",type=str)
    parser.add_argument("--batch_size",type=int)    
    parser.add_argument("--lazy",action='store_true')
    args = parser.parse_args()
    return args    

if __name__ == "__main__":
    args=parse_args(default_conf='conf/l1.cfg')
    conf_dict=conf.read_conf(args.conf,['clf'])
    conf.add_dir_paths(conf_dict,args.data_dir,
                        args.main_dir)
    conf.GLOBAL['batch_size']=args.batch_size
    print(conf_dict)
    conf_dict['lazy']=args.lazy
    test_exp(conf_dict) 