import os,json
from sklearn.metrics import accuracy_score

def metric_dict(metric_type,pred_path):
    metric_i=get_metric(metric_type)
    metric_dict={}
    for path_i in top_files(pred_path):
        all_pred=read_pred(path_i)
        id_i=get_id(path_i)
        line_i=[id_i]
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

def get_metric(metric_type):
    if(metric_type=='acc'):
        return accuracy_score
    return metric_type