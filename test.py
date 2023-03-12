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
import numpy as np
import json
import conf,data,learn,utils,ens_feats
from tqdm import tqdm

def test_exp(conf):
    logging.basicConfig(filename='{}_test.log'.format(conf['log']), 
        level=logging.INFO,filemode='w', 
        format='%(process)d-%(levelname)s-%(message)s')
    @utils.dir_fun(as_dict=True)
    def helper(data_i):
        model_i='%s/%s' % (conf['model'],data_i.split('/')[-1])
        return exp(data_i,model_i,conf)
    logging.info('Save results:%s' % conf['result'])
    lines_dict=helper(conf['json'])

def exp(data_path,model_path,conf):
    raw_data=data.read_data(data_path)
    clf_types,ens_types=conf['clf_types'],conf['ens_types']
    print(f'Test models on dataset:{data_path}')
    print('Multiclass classifiers types used:{}'.format(','.join(clf_types)))
    print('Ensemble variant used:{}'.format(','.join(ens_types)))
    helper= get_fold_fun(raw_data,clf_types,ens_types)
    acc=[helper(path_i) 
        for path_i in tqdm(data.top_files(model_path))]
    acc_dict={ id_i:[] for id_i in acc[0].keys()}
    for acc_i in acc:
        for key_j,value_j in acc_i.items():
            acc_dict[key_j].append(value_j) 
    data_i=data_path.split('/')[-1]    
    with open(conf['result'],"a") as f:
        f.write('dataset,ens_type,clf_type,mean_acc,std_acc\n')
    for id_j,acc_j in acc_dict.items():
        line_j=f'{data_i},{id_j},{stats(acc_j)}'
        with open(conf['result'],"a") as f:
            f.write(line_j+ '\n')
    print('Saved result for {} at {}\n'.format(data_path,conf['result']))

def get_fold_fun(raw_data,clf_types,ens_types):
    ens_types=[ens_feats.get_ensemble(type_i)
        for type_i in ens_types]
    @utils.unify_cv(dir_path=None,show=False)
    def helper(in_path):
        logging.info(f'Read models:{in_path}')
        common,binary= gen_feats(raw_data,in_path)
        acc_dir={}
        for ens_i in ens_types:
            for clf_j in clf_types:
                ens_inst=ens_i(common,binary)
                id_ij=f'{str(ens_inst)},{clf_j}'
                acc_dir[id_ij]=ens_inst(clf_j)
        return acc_dir
    return helper

def gen_feats(raw_data,in_path):
    model_path=f'{in_path}/models'
    models=[keras.models.load_model(path_i,compile=False)
        for path_i in data.top_files(model_path)]  
    rename_path=f'{in_path}/rename'
    with open(rename_path, 'r') as f:
        rename_dict= json.load(f)
    common=raw_data.rename(rename_dict)
    X,y,names=common.as_dataset()
    binary=[]
    for model_i in models:
        X_i=model_i.predict(X)
        binary_i=data.from_names(X_i,names)
        binary.append(binary_i)
    return common,binary

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