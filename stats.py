import  numpy as np
import os
from tensorflow import keras
import time
from collections import defaultdict
import test,utils,data,ens_feats

def variant_time(data_path,model_path):
    clf_types=['SVC']#'MLP-TF','RF']#,'LR','LR-imb',]
    ens_types=['base','binary','common']
    raw_data=data.read_data(data_path)
    common,binary= test.gen_feats(raw_data,model_path)
    lines=[]
    for ens_i in ens_types:
        for clf_j in clf_types:
            st=time.time()
            ens_inst=ens_feats.get_ensemble(ens_i)(common,binary)
            id_ij=f'{str(ens_inst)},{clf_j}'
            ens_inst(clf_j)
            lines.append(f'{id_ij},{(time.time()-st):.4f}s')
            print(lines[-1])
    print('\n'.join(lines))

def full_time(in_path):
    parse_time(in_path,'info_train.log','Save')
    parse_time(in_path,'info_test.log','Evaluate')

def parse_time(in_path,filename='info_train.log',line_type='Save'):
    train_path=f'{in_path}/{filename}'
    time_dict = defaultdict(lambda :[])
    with open(train_path) as f:
        lines=f.readlines()
        for line_i in lines:
            if(line_i.find(line_type)> -1):
                raw_i=line_i.split('/')
                name_i=raw_i[-3]
                time_i= raw_i[-1].split(' ')[-1]
                time_i=time_i.replace('s','')
                time_dict[name_i].append(float(time_i))
    stats_from_dict(time_dict)

def stats_from_dict(time_dict):
    for name_i,time_i in time_dict.items():
        stats= np.mean(time_i),np.std(time_i)
        print(f'{name_i},{stats[0]:.4},{stats[1]:.4}')

def show_dim(in_path):
    @utils.dir_fun(as_dict=True)
    def helper(in_path):
        model_path=f'{in_path}/0/0'
        rename_dict,models=test.read_fold(model_path)
        dim=models[0].output.shape[-1]
        return dim
    dim_dict=helper(f'{in_path}/models')
    txt= to_txt(dim_dict)
    print(txt)
    return txt

#def show_size(in_path):
#    stats=size_stats(in_path)
#    for name_i,stat_i in stats.items():
#        print(f'{name_i}:{stat_i[0]:.2}')

#@utils.dir_fun(True)
#def size_stats(in_path):
#    sizes=[get_size(path_i) 
#        for path_i in data.top_files(in_path)] 
#    return np.mean(sizes),np.std(sizes)

#def get_size(in_path):
#    total_size=0
#    for root, dir, files in os.walk(in_path):
#        for file_i in files:
#    	    path_i=f"{root}/{file_i}"
#    	    size=os.path.getsize(path_i)
#    	    total_size+=size
#    return total_size

def to_txt(result_dict):
    lines=[ f'{id_i},{str(result_i)}' 
        for id_i,result_i in result_dict.items()]
    return '\n'.join(lines)

#variant_time(data_path,model_path)
#full_time('../small/ovo')
show_dim('../small/hyper')