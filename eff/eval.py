import tools
tools.silence_warnings()
import numpy as np
import pandas as pd
import os,argparse
from collections import defaultdict
import pred,tools

class MetricDict(defaultdict):
    def __init__(self):#, arg=[]):
        super(MetricDict, self).__init__(lambda :[])
    
    def to_df(self):
        lines=[]
        for name_i,acc_i in self.items():
            line_i=name_i.split('-')
            line_i+=[np.mean(acc_i),np.std(acc_i)]
            lines.append(line_i)
        cols=['ens','cs','clf','mean','std']
        df=pd.DataFrame(lines,columns=cols)
        df=df.sort_values(by='mean',
                              ascending=False)
        print(df)

    def sizes(self):
        for name_i,acc_i in self.items():
            print(len(acc_i) )	


def read_metric_dict(pred_path):
    metric=tools.get_metric('acc')
    metric_dict=MetricDict() #defaultdict(lambda :[])
    for path_i in all_files(pred_path):
        id_i=get_id(path_i)
        for pred_j,test_j in pred.read_pred(path_i):
            acc_j=metric(pred_j,test_j)
            metric_dict[id_i].append(acc_j)
    return metric_dict

def all_files(in_path):
    for folder, subfolders, files in os.walk(in_path):
        for file_i in files:
            path_i = os.path.abspath(os.path.join(folder, file_i))
            yield path_i

def get_id(path_i):
    raw=path_i.split('/')
    return f'{raw[-3]}-{raw[-1]}'


def is_pred_dir(dir_path):
    for path_i in tools.top_files(dir_path):
        if(not path_i.split('/')[-1].isnumeric()):
            return False
    return True

def eval_exp(pred_path):
    def helper(pred_path):
        print(pred_path)
        metric_dict= read_metric_dict(pred_path)
        metric_dict.to_df()
    print(is_pred_dir(pred_path))
#    if(os.path.isdir(pred_path)):
#        helper=tools.dir_fun(3)(helper)
#    helper(pred_path)

if __name__ == '__main__':
    dir_path='../../s_10_10'
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default=f'{dir_path}/pred/binary/cleveland')
    args = parser.parse_args()
    eval_exp(args.pred)
