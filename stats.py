import  numpy as np
import os
from tensorflow import keras
import time
#import binary,data,utils,train
import test,utils,data,ens_feats

def variant_time(data_path,model_path):
    clf_types=['LR','LR-imb','RF']
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

def show_dim(in_path):
    @utils.dir_fun(as_dict=True)
    def helper(in_path):
        model_path=f'{in_path}/0/0/models/0'
        nn=keras.models.load_model(model_path)
        return nn.output.shape[1:][0]
    dim_dict=helper(in_path)
    return to_txt(dim_dict)

def show_size(in_path):
    stats=size_stats(in_path)
    for name_i,stat_i in stats.items():
        print(f'{name_i}:{stat_i[0]:.2}')

@utils.dir_fun(True)
def size_stats(in_path):
    sizes=[get_size(path_i) 
        for path_i in data.top_files(in_path)] 
    return np.mean(sizes),np.std(sizes)

def get_size(in_path):
    total_size=0
    for root, dir, files in os.walk(in_path):
        for file_i in files:
    	    path_i=f"{root}/{file_i}"
    	    size=os.path.getsize(path_i)
    	    total_size+=size
    return total_size

def to_txt(result_dict):
    lines=[ f'{id_i},{str(result_i)}' 
        for id_i,result_i in result_dict.items()]
    return '\n'.join(lines)

data_path='../slow/json/mfeat-fourier'
model_path= '../slow/models/mfeat-fourier/0/0'
variant_time(data_path,model_path)