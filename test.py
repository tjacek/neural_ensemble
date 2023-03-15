#from pathlib import Path
#sys.path.append(str(Path('.').absolute().parent))
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
    if(os.path.isdir(conf['output'])):
        shutil.rmtree(conf['output'])
    if(os.path.exists(conf['result'])):
        os.remove(conf['result'])
    logging.basicConfig(filename='{}_test.log'.format(conf['log']), 
        level=logging.INFO,filemode='w', 
        format='%(process)d-%(levelname)s-%(message)s')
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
                id_ij=f'{str(ens_inst)}_{clf_j}'
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

def make_results(conf):
    @utils.dir_fun(True)
    @utils.dir_fun(True)
    def helper(in_path):
        acc=[learn.read_result(path_i).get_acc()
           for path_i in data.top_files(in_path) ]
        return acc
    acc_dict=helper(conf['output'])
    with open(conf['result'],"a") as f:
        f.write('dataset,ens_type,clf_type,mean_acc,std_acc,max_acc\n')
        for data_i,dict_i in acc_dict.items():
            for id_j,acc_j in dict_i.items():
                line_ij=f'{data_i},{id_j},{stats(acc_j)}'
                f.write(line_ij+ '\n')

def stats(acc,as_str=True):
    raw=[f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std,np.amax]]
    if(as_str):
        return ','.join(raw)
    return raw

def best_frame(result_path,out_path=None):
    result=pd.read_csv(result_path)
    lines=[]
    for data_i in result['dataset'].unique():
        df_i=result[result['dataset']==data_i]
        k=df_i['mean_acc'].argmax()
        row_i=df_i.iloc[k].to_list()
        lines.append(row_i)
    best=pd.DataFrame(lines,columns=result.columns)
    best.to_csv(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",type=str,default='conf/base.cfg')
    args = parser.parse_args()
    conf_dict=conf.read_test(args.conf)
    test_exp(conf_dict)
    make_results(conf_dict)
    print("Saved results at {}".format(conf_dict['result']))
    best_frame(conf_dict['result'],conf_dict['best'])
    print("Saved best results at {}".format(conf_dict['best']))