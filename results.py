import conf 
conf.silence_warnings()
import os,argparse
import numpy as np
import pandas as pd
import shutil
import data,learn,utils

def make_results(conf):
    @utils.dir_fun(True)
    @utils.dir_fun(True)
    def helper(in_path):
        acc=[learn.read_result(path_i).get_acc()
           for path_i in data.top_files(in_path) ]
        return acc
    acc_dict=helper(conf['output'])
    if(os.path.isdir(conf_dict['result'])):
        shutil.rmtree(conf_dict['result'])
    with open(conf['result'],"a") as f:
        f.write('dataset,ens_type,clf_type,mean_acc,std_acc,max_acc\n')
        for data_i,dict_i in acc_dict.items():
            for id_j,acc_j in dict_i.items():
                id_j=','.join(id_j.split('_'))
                line_ij=f'{data_i},{id_j},{stats(acc_j)}'
                f.write(line_ij+ '\n')

def stats(acc,as_str=True):
    raw=[f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std,np.amax]]
    if(as_str):
        return ','.join(raw)
    return raw

def best_frame(result_path,out_path=None):
    result=pd.read_csv(result_path)
    lines=[]
    for data_i in result['dataset'].unique():
        df_i=result[result['dataset']==data_i]
        k=df_i['mean_acc'].argmax()
        row_i=df_i.iloc[k].to_list()
        lines.append(row_i)
    best=pd.DataFrame(lines,columns=result.columns)
    best.to_csv(out_path)

def parse_args(default_conf='conf/l1.cfg'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf",type=str,default=default_conf)
#    parser.add_argument("--data_dir",type=str)
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