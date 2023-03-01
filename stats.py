import  numpy as np
import os
from tensorflow import keras

import data,utils

def show_dim(in_path):
    @utils.dir_fun(as_dict=True)
    def helper(in_path):
        model_path=f'{in_path}/0/0/models/0'
        nn=keras.models.load_model(model_path)
        return nn.output.shape[1:][0]
    dim_dict=helper(in_path)
    lines=[ f'{id_i},{dim_i}' 
        for id_i,dim_i in dim_dict.items()]
    return '\n'.join(lines)

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

in_path='../uci_npz/uci_10-fold/'
txt=show_dim('test')
print(txt)