import conf 
conf.silence_warnings()
import os,argparse
import numpy as np
import pandas as pd
from itertools import product
import shutil
import collections
import data,learn,utils,output.cf,output.plot

class Metrics(object):
    def __init__(self):
        metrics=[('acc',learn.acc_metric),('precision',learn.precision_metric),
                ('recall',learn.recall_metric), ('f1',learn.f1_metric)]
        stats=[('mean',np.mean),('std',np.std)]#,('max',np.amax)]
        self.metrics_dict= collections.OrderedDict(metrics)
        self.stats_dict= collections.OrderedDict(stats)
    
    def get_cols(self):
        cols=[]
        for metric_i in self.metrics_dict.keys():
            for stats_i in self.stats_dict.keys():
                cols.append(f'{metric_i}_{stats_i}')
        return ','.join(cols) 

    def __call__(self,results):
        line_i=[]
        for id_i,metric_i in self.metrics_dict.items():
            values=[]
            for result_k in results:
                values.append(metric_i(result_k))
            for id_j,stats_j in self.stats_dict.items():
                line_i.append(f'{stats_j(values):.4f}')
        return ','.join(line_i)

def make_results(conf,metrics=None):
    if(metrics is None):
        metrics=Metrics()
    if(os.path.exists(conf['result'])):
        os.remove(conf['result'])
    @utils.dir_fun(True)
    @utils.dir_fun(True)
    def helper(in_path):
        result=[learn.read_result(path_i)
           for path_i in data.top_files(in_path) ]
        return result
    result_dict=helper(conf['output'])
    if(os.path.isdir(conf_dict['result'])):
        shutil.rmtree(conf_dict['result'])
    with open(conf['result'],"a") as f:
        f.write(f'dataset,ens_type,clf_type,{metrics.get_cols()}\n') 
        for data_i,dict_i in result_dict.items():
            for id_j,result_j in dict_i.items():
                id_j=id_j.replace('_',',')
                line_ij=f'{data_i},{id_j},{metrics(result_j)}\n'
                f.write(line_ij)

def best_frame(result_path,out_path=None):
    result=pd.read_csv(result_path)
    lines=[]
    for data_i in result['dataset'].unique():
        df_i=result[result['dataset']==data_i]
        k=df_i['acc_mean'].argmax()
        row_i=df_i.iloc[k].to_list()
        lines.append(row_i)
    best=pd.DataFrame(lines,columns=result.columns)
    best.to_csv(out_path)

def parse_args(default_conf='conf/l1.cfg'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",type=str,default=default_conf)
    parser.add_argument("--main_dir",type=str)
    args = parser.parse_args()
    return args    

if __name__ == "__main__":
    args=parse_args(default_conf='conf/l1.cfg')
    conf_dict=conf.read_conf(args.conf,['clf'])
    conf.add_dir_paths(conf_dict,None,#args.data_dir,
                        args.main_dir)
    make_results(conf_dict)
    print("Saved results at {}".format(conf_dict['result']))
    best_frame(conf_dict['result'],conf_dict['best'])
    print("Saved best results at {}".format(conf_dict['best']))
    output.cf.gen_cf(conf_dict ,sep='_')
    print("Saved confusion matrix at {}".format(conf_dict['cf']))
    output.plot.box_gen(conf_dict)
    print("Saved box plot at {}".format(conf_dict['box']))