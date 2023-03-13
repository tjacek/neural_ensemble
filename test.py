#from pathlib import Path
#sys.path.append(str(Path('.').absolute().parent))
import os
import sys
import logging,argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from tensorflow import keras
from keras.models import model_from_json
import numpy as np
import json,time,gzip,shutil
import conf,data,learn,utils,ens_feats
from tqdm import tqdm

def test_exp(conf):
    if(os.path.isdir(conf['output'])):
        shutil.rmtree(conf['output'])
    logging.basicConfig(filename='{}_test.log'.format(conf['log']), 
        level=logging.INFO,filemode='w', 
        format='%(process)d-%(levelname)s-%(message)s')
#    @utils.dir_fun(as_dict=True)
    @utils.dir_map(1)
    def helper(model_path,output_path):
        name=model_path.split('/')[-1]
        data_path='{}/{}'.format(conf['json'],name)
        return exp(data_path,model_path,output_path,conf)
#    logging.info('Save results:%s' % conf['result'])
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

#    full_result={ id_i:[] for id_i in result[0].keys()}    
#    data_i=data_path.split('/')[-1]    
#    with open(conf['result'],"a") as f:
#        f.write('dataset,ens_type,clf_type,mean_acc,std_acc,max_acc\n')
#        for id_j,acc_j in acc_dict.items():
#            raise Exception(type(acc_j[0]))
#            line_j=f'{data_i},{id_j},{stats(acc_j)}'
#        with open(conf['result'],"a") as f:
#            f.write(line_j+ '\n')

def get_fold_fun(raw_data,clf_types,ens_types):
    ens_types=[ens_feats.get_ensemble(type_i)
        for type_i in ens_types]
    @utils.unify_cv(dir_path=None,show=False)
    def helper(in_path):
        st=time.time()
        logging.info(f'Read models:{in_path}')
        common,binary= gen_feats(raw_data,in_path)
        acc_dir={}
        for ens_i in ens_types:
            for clf_j in clf_types:
                ens_inst=ens_i(common,binary)
                id_ij=f'{str(ens_inst)},{clf_j}'
                acc_dir[id_ij]=ens_inst(clf_j)
        logging.info(f'Evaluate models from:{in_path} took {(time.time()-st):.4f}s')
        return acc_dir
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

def stats(acc,as_str=True):
    raw=[f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std,np.amax]]
    if(as_str):
        return ','.join(raw)
    return raw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",type=str,default='conf/base.cfg')
    args = parser.parse_args()
    conf_dict=conf.read_test(args.conf)
    test_exp(conf_dict)