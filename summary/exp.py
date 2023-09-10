import numpy as np
import pandas as pd
import utils

SEP='|'

class MetricDict(dict):
    def __init__(self, arg=[]):
        super(MetricDict, self).__init__(arg)
        self.cols=['dataset','cs','clf','variant']
    
    def to_df(self,taboo):
        lines=[]
        stats=[np.mean,np.std]
        for id_i,metric_i in self.items():
            if(id_i.is_valid(taboo)):
                line_i=id_i.split('|')
                line_i+=[round(stat_j(metric_i),4) 
                        for stat_j in stats]
                line_i.append(id_i)
                lines.append(line_i)
        cols=self.cols+['mean','std','id']
        df=pd.DataFrame(lines,columns=cols)
        return df

class ExpID(str):
    def __getitem__(self, key):
        return self.split(SEP)[key]
    
    def is_valid(self,taboo):
        raw=self.split(SEP)
        for i,taboo_i in enumerate( taboo):
            for taboo_j	in taboo_i:
                if(raw[i]==taboo_j):
                    return False	
        return True

def read_acc(pred_path,metric_type='acc'):
    metric_i=utils.get_metric(metric_type)
    acc_dict=MetricDict()
    for path_i in utils.top_files(pred_path):
        data_i=path_i.split('/')[-1]
        for path_j in utils.top_files(path_i):
            id_i=path_j.split('/')[-1]
            if('base' in id_i):
                exp_id=[data_i,'-',id_i.split('-')[-1],'-']
            else:
                exp_id=[data_i]+id_i.split('-')
            exp_id=ExpID('|'.join(exp_id)) 	
            all_pred=utils.read_pred(path_j)
            acc_ij=[ metric_i(test_i,pred_i) 
                       for test_i,pred_i in all_pred]
            acc_dict[exp_id]=acc_ij
    return acc_dict

dir_path='../../pred'
acc_dict= read_acc(dir_path)

taboo=[[],['multi'],['SVC'],['cs','inliner']]
df=acc_dict.to_df(taboo)
print(df)