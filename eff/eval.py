import tools
tools.silence_warnings()
import numpy as np
import pandas as pd
import os,argparse
from collections import defaultdict
import pred

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default=f'pred')
    args = parser.parse_args()
    metric_dict= read_metric_dict(args.pred)
    metric_dict.to_df()