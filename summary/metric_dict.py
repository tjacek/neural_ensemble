import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score

class MetricDict(dict):
    def __init__(self, arg=[]):
        super(MetricDict, self).__init__(arg)

    def to_df(self):
        lines=[]
        stats=[np.mean,np.std]
        for id_i,metric_i in self.items():
            line_i=id_i.split(',')
            line_i+=[round(stat_j(metric_i),4) 
                        for stat_j in stats]
            lines.append(line_i)
        cols=['dataset','clf','mean','std']
        return pd.DataFrame(lines,columns=cols)

def make_acc_dict(pred_path,metric_i='acc'):
    if(metric_i=='acc'):
        metric_i=accuracy_score
    elif(metric_i=='balanced'):
        metric_i=balanced_accuracy_score
    metric_dict=MetricDict()
    for path_i in top_files(pred_path):
        all_pred=read_pred(path_i)
        id_i=get_id(path_i)
        line_i=[get_id(path_i)]
        acc=[ metric_i(test_i,pred_i) 
                for test_i,pred_i in all_pred]
        metric_dict[id_i]=acc
    return metric_dict

def read_pred(path_i):
    with open(path_i, 'r') as f:        
        json_bytes = f.read()                      
        return json.loads(json_bytes)

def top_files(path):
    if(type(path)==str):
        paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    else:
        paths=path
    paths=sorted(paths)
    return paths

def get_id(path_i):
    raw=path_i.split('/')
    return f'{raw[-2]},{raw[-1]}'