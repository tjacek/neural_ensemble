import  numpy as np
import os
import data,utils

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
stats=size_stats(in_path)
for name_i,stat_i in stats.items():
    print(f'{name_i}:{stat_i[0]:.2}')#,{stat_i[1]:.2}')