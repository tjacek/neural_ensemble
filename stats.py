import  numpy as np
import os
from tensorflow import keras
import binary,data,utils,train

def show_bayes(in_path, out_path,n_split=3):
    NeuralEnsemble=binary.get_ens('all')
    f=open(out_path,'w')
    @utils.dir_fun(as_dict=False)#True)
    def helper(in_path):    
        raw_data=data.read_data(in_path)
        hyper=train.find_hyperparams(raw_data,
            ensemble_type=NeuralEnsemble,n_split=n_split)
        name_i=in_path.split('/')[-1]
        line=f'{name_i},{str(hyper)}\n'
        f.write(line)
        return hyper
    result=helper(in_path)

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

txt=show_bayes('small','bayes')
