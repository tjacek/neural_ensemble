import os,re
import json
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from sklearn.metrics import accuracy_score,balanced_accuracy_score

class MetricDict(dict):
    def __init__(self, arg=[]):
        super(MetricDict, self).__init__(arg)

    def to_df(self,drop=False,transform=None):
        lines=[]
        stats=[np.mean,np.std]
        for id_i,metric_i in self.items():
            line_i=id_i.split(',')
            line_i+=[round(stat_j(metric_i),4) 
                        for stat_j in stats]
            lines.append(line_i)
        cols=['dataset','id','mean','std']
        df=pd.DataFrame(lines,columns=cols)
        col_dict={ 'clf': GenCol(['RF','SVC','LR']),
                    'cs': GenCol(['weighted','binary','multi']),
               'variant': GenCol(['necscf','cs','inliner'])}
#                  'alpha':get_alpha}
        for col_i,fun_i in col_dict.items():
            df[col_i]=df['id'].apply(fun_i)
        if(drop):
            df.drop('id', inplace=True, axis=1)
        if(not (transform is None)):
            df=transform(df)
        return df
    
    def dataset_dfs(self,drop=False,transform=None):
        df=self.to_df(drop=drop,transform=transform)
        data_dict={}
        for data_i in df['dataset'].unique():
            data_dict[data_i]= df[ df['dataset']==data_i]
        return data_dict

    def by_clf(self,drop=False,transform=None):
        clf_dict=defaultdict(lambda:{})
        for name_i,data_i in self.dataset_dfs(drop,transform).items():
            for clf_j in data_i['clf'].unique():
                clf_dict[name_i][clf_j]=data_i[data_i['clf']==clf_j]
        return clf_dict

    def pvalue(self,id_x,id_y):
        r=stats.ttest_ind(self[id_x], 
                self[id_y], equal_var=False)
        p_value=round(r[1],4)
        return p_value

    def stats(self):
        return {key_i:len(value_i) 
                for key_i,value_i in self.items()}

class GenCol(object):
    def __init__(self,seqs,default='-'):
        self.seqs=seqs
        self.default=default

    def __call__(self,id_i):
        for seq_i in self.seqs:
            if(seq_i in id_i):
                return seq_i
        return self.default

class AccDictReader(object):
    def __init__(self,taboo):
        self.taboo=taboo

    def __call__(self,pred_path,metric_i='acc'):
        metric_i=get_metric(metric_i)
        metric_dict=MetricDict()
        for path_i in top_files(pred_path):
            all_pred=read_pred(path_i)
            id_i=get_id(path_i)
            if(self.valid(id_i)):
                line_i=[get_id(path_i)]
                acc=[ metric_i(test_i,pred_i) 
                    for test_i,pred_i in all_pred]
                metric_dict[id_i]=acc
        return metric_dict
    
    def valid(self,id_i):
        for taboo_j in self.taboo:
            if(taboo_j in id_i):
                return False
        return True 

def get_metric(metric_i):
    if(metric_i=='acc'):
        return accuracy_score
    elif(metric_i=='balanced'):
        return balanced_accuracy_score

def get_alpha(raw):
    digits=re.findall(r'\d+',raw)
    if(len(digits)>0):
        return f'0.{digits[1]}'
    return '-'

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
    data_i=raw[-2].replace('-','_')
    return f'{data_i},{raw[-1]}'